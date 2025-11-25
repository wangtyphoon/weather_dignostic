import csv
import datetime as dt
import math
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt


EARTH_RADIUS_M = 6371000.0


def parse_best_track(path: Path):
    """Parse the best-track style txt file, forward-filling dates."""
    records = []
    current_date = None
    prev_dt = None

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row or row[0].startswith(("BASIN", "degrees north")):
                continue
            if len(row) < 7:
                continue

            time_field = row[1].strip()
            if "-" in time_field:
                timestamp = dt.datetime.strptime(time_field, "%Y-%m-%d %H:%M:%S")
                current_date = timestamp.date()
            else:
                if current_date is None:
                    raise ValueError("Encountered HH:MM:SS without a known date")
                time_only = dt.datetime.strptime(time_field, "%H:%M:%S").time()
                timestamp = dt.datetime.combine(current_date, time_only)
                if prev_dt and timestamp < prev_dt:
                    # Handle a day rollover if the date is omitted.
                    timestamp += dt.timedelta(days=1)
                    current_date = timestamp.date()

            lat = float(row[3])
            lon = float(row[4])
            nature = row[2].strip()

            records.append(
                {
                    "time": timestamp,
                    "lat": lat,
                    "lon": lon,
                    "nature": nature,
                    "wind": float(row[5]),
                    "pressure": float(row[6]),
                    "speed_kmh": None,
                    "speed_kt": None,
                }
            )
            prev_dt = timestamp

    if not records:
        raise ValueError("No records parsed from the track file")
    return records


def saffir_simpson_category(wind_kt: float):
    """Return category string and color by US hurricane scale."""
    if wind_kt < 34:
        return "TD/Dep", "#5dade2"
    if wind_kt < 64:
        return "TS", "#1abc9c"
    if wind_kt < 83:
        return "Cat1", "#f1c40f"
    if wind_kt < 96:
        return "Cat2", "#e67e22"
    if wind_kt < 113:
        return "Cat3", "#e74c3c"
    if wind_kt < 137:
        return "Cat4", "#c0392b"
    return "Cat5", "#8e44ad"


def append_translation_speed(records):
    """Compute translation speed between consecutive fixes."""
    for first, second in zip(records, records[1:]):
        dlon = math.radians(second["lon"] - first["lon"])
        dlat = math.radians(second["lat"] - first["lat"])
        lat_mean = math.radians((first["lat"] + second["lat"]) * 0.5)
        dx = EARTH_RADIUS_M * dlon * math.cos(lat_mean)
        dy = EARTH_RADIUS_M * dlat
        dt_seconds = (second["time"] - first["time"]).total_seconds()
        speed_mps = math.hypot(dx, dy) / dt_seconds
        second["speed_kmh"] = speed_mps * 3.6
        second["speed_kt"] = speed_mps * 1.94384


def plot_track(records, output_path: Path):
    append_translation_speed(records)

    lats = [r["lat"] for r in records]
    lons = [r["lon"] for r in records]
    winds = [r["wind"] for r in records]
    speeds_kmh = [r["speed_kmh"] for r in records]
    speeds_kt = [r["speed_kt"] for r in records]
    times = [r["time"] for r in records]

    fig = plt.figure(figsize=(13, 6))
    proj = ccrs.PlateCarree()
    ax_map = fig.add_subplot(1, 2, 1, projection=proj)
    ax_speed = fig.add_subplot(1, 2, 2)

    ax_map.add_feature(cfeature.COASTLINE.with_scale("50m"))
    ax_map.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.5, linestyle=":")
    gl = ax_map.gridlines(draw_labels=True, linewidth=0.5, alpha=0.4)
    gl.right_labels = False
    gl.top_labels = False

    # Plot path and category-colored markers.
    ax_map.plot(lons, lats, "-", color="gray", lw=1, transform=proj, zorder=1)
    handles = {}
    for lon, lat, wind in zip(lons, lats, winds):
        cat, color = saffir_simpson_category(wind)
        point = ax_map.scatter(
            lon,
            lat,
            s=30,
            color=color,
            edgecolors="k",
            linewidths=0.4,
            transform=proj,
            zorder=2,
        )
        handles.setdefault(cat, point)

    start_handle = ax_map.scatter(lons[0], lats[0], marker="*", s=120, color="gold", edgecolors="k", transform=proj, zorder=3, label="Start")
    end_handle = ax_map.scatter(lons[-1], lats[-1], marker="X", s=70, color="black", edgecolors="white", transform=proj, zorder=3, label="End")

    # 標註路徑上每隔兩天的日期。
    label_interval = dt.timedelta(days=2)
    last_labeled_date = times[0].date() - label_interval
    for lon, lat, when in zip(lons, lats, times):
        current_date = when.date()
        if current_date - last_labeled_date >= label_interval:
            ax_map.text(
                lon + 0.2,
                lat + 0.2,
                current_date.strftime("%m/%d"),
                fontsize=9,
                weight="bold",
                color="black",
                transform=proj,
                zorder=4,
                bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "alpha": 0.7, "linewidth": 0},
            )
            last_labeled_date = current_date

    # Legend for categories.
    cat_order = ["TD/Dep", "TS", "Cat1", "Cat2", "Cat3", "Cat4", "Cat5"]
    cat_handles = [handles[c] for c in cat_order if c in handles]
    cat_labels = [c for c in cat_order if c in handles]
    ax_map.legend(cat_handles + [start_handle, end_handle], cat_labels + ["Start", "End"], loc="upper right", fontsize=8)
    ax_map.set_title("13L Track (Cartopy) with US Hurricane Categories")

    lon_pad = 2
    lat_pad = 2
    ax_map.set_extent([min(lons) - lon_pad, max(lons) + lon_pad, min(lats) - lat_pad, max(lats) + lat_pad], crs=proj)

    # Speed time series.
    times_speed = times[1:]
    speeds_kmh_valid = speeds_kmh[1:]
    winds_valid = winds[1:]

    ax_speed.plot(times_speed, speeds_kmh_valid, "-o", color="#005f9e", ms=3, label="Speed (km/h)")
    ax_speed.set_ylabel("Translation speed (km/h)")
    ax_speed.set_title("Translation speed & intensity vs time")
    ax_speed.grid(True, linestyle="--", alpha=0.4)
    ax_speed.tick_params(axis="x", rotation=45)

    ax_wind = ax_speed.twinx()
    ax_wind.plot(times_speed, winds_valid, "-s", color="#e67e22", ms=3, label="Intensity (kt)")
    ax_wind.set_ylabel("Intensity (kt)")

    # Combined legend
    lines1, labels1 = ax_speed.get_legend_handles_labels()
    lines2, labels2 = ax_wind.get_legend_handles_labels()
    ax_speed.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    track_file = Path("13L_best_track.txt")
    output_path = Path("13L_track.png")
    records = parse_best_track(track_file)
    plot_track(records, output_path)

    print(f"生成圖檔: {output_path}")
    print("各時間點平移速度 (km/h, kt):")
    for r in records[1:]:
        print(f"{r['time']}: {r['speed_kmh']:.1f} km/h, {r['speed_kt']:.1f} kt")


if __name__ == "__main__":
    main()

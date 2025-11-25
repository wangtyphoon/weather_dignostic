import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Paths
THETA_PATH = "13L_thetao_0-500m_20251023-29.nc"
BEST_TRACK_PATH = "13L_best_track.txt"

# Constants
RHO = 1025.0  # kg/m3
CP = 3994.0   # J/(kg*K)


def parse_best_track(path):
    """
    Parse best-track file and return list of (datetime64[h], lat, lon, wind_kt).
    Handles rows with full date+time and time-only rows that inherit the last date.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    out = []
    current_date = None
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()[2:]  # skip headers

    for line in lines:
        if not line.strip():
            continue
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        dt = None
        lat_idx = None

        # Case: full date and time in two columns (e.g., 2025-10-23 00:00:00)
        if len(parts) >= 8 and "-" in parts[1] and ":" in parts[2]:
            date_str, time_str = parts[1], parts[2]
            dt = np.datetime64(f"{date_str}T{time_str}").astype("datetime64[h]")
            current_date = dt.astype("datetime64[D]")
            lat_idx = 4
        # Case: time only, use current_date
        elif ":" in parts[1] and current_date is not None:
            time_str = parts[1]
            dt = np.datetime64(f"{str(current_date)}T{time_str}").astype("datetime64[h]")
            lat_idx = 3

        if dt is None or lat_idx is None:
            continue

        try:
            lat_val = float(parts[lat_idx])
            lon_val = float(parts[lat_idx + 1])
            wind = float(parts[lat_idx + 2]) if len(parts) > lat_idx + 2 else np.nan
        except (ValueError, IndexError):
            continue

        out.append((dt, lat_val, lon_val, wind))

    return out


def compute_layer_thickness(depth_vals):
    """Midpoint thickness for 1D depth array."""
    edges = np.zeros(depth_vals.size + 1)
    edges[1:-1] = (depth_vals[:-1] + depth_vals[1:]) / 2
    edges[0] = depth_vals[0] - (depth_vals[1] - depth_vals[0]) / 2
    edges[-1] = depth_vals[-1] + (depth_vals[-1] - depth_vals[-2]) / 2
    return np.diff(edges)


def main():
    # Load theta
    theta = xr.open_dataset(THETA_PATH)["thetao"]
    lon = theta.longitude
    lat = theta.latitude

    # Precompute thickness for OHC (J/m^2)
    thick = compute_layer_thickness(theta.depth.values)
    thick_da = xr.DataArray(thick, coords={"depth": theta.depth}, dims=["depth"])

    # Parse track and keep times inside dataset range
    track = parse_best_track(BEST_TRACK_PATH)
    t_min = theta.time.min().values.astype("datetime64[h]")
    t_max = theta.time.max().values.astype("datetime64[h]")
    track = [p for p in track if t_min <= p[0] <= t_max]
    if not track:
        raise ValueError("No track points within dataset time range.")

    times = []
    sst_list = []
    tchp_list = []
    wind_list = []

    for dt, lat_pt, lon_pt, wind in track:
        da_time = theta.sel(time=dt, method="nearest")
        sst = da_time.sel(depth=theta.depth.min(), method="nearest").sel(
            latitude=lat_pt, longitude=lon_pt, method="nearest"
        ).item()

        # TCHP relative to 26°C (kJ/cm^2)
        excess = (da_time - 26).where(da_time > 26, 0)
        tchp = (excess.sel(latitude=lat_pt, longitude=lon_pt, method="nearest") * thick_da).sum("depth") * RHO * CP * 1e-7

        times.append(dt)
        sst_list.append(sst)
        tchp_list.append(tchp.item())
        wind_list.append(wind)

    # Prepare arrays for plotting
    times_arr = np.array(times)
    lats_arr = np.array([p[1] for p in track])
    lons_arr = np.array([p[2] for p in track])
    wind_arr = np.array(wind_list)

    # Plot track (left) + time series (right) together
    fig = plt.figure(figsize=(13, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.1])

    # Map
    ax_map = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    ax_map.add_feature(cfeature.LAND, facecolor="lightgray")
    ax_map.add_feature(cfeature.COASTLINE, linewidth=0.6)
    ax_map.add_feature(cfeature.BORDERS, linewidth=0.4)
    gl = ax_map.gridlines(draw_labels=True, linewidth=0.3, alpha=0.6, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    # zoom to track with a small margin
    lon_pad = 2.0
    lat_pad = 2.0
    ax_map.set_extent(
        [
            float(lons_arr.min()) - lon_pad,
            float(lons_arr.max()) + lon_pad,
            float(lats_arr.min()) - lat_pad,
            float(lats_arr.max()) + lat_pad,
        ],
        crs=ccrs.PlateCarree(),
    )

    sc = ax_map.scatter(lons_arr, lats_arr, c=wind_arr, cmap="magma", s=35, transform=ccrs.PlateCarree(), zorder=3)
    ax_map.plot(lons_arr, lats_arr, color="k", linewidth=1.0, transform=ccrs.PlateCarree(), zorder=2)
    ax_map.plot(lons_arr[0], lats_arr[0], marker="^", color="green", markersize=8, transform=ccrs.PlateCarree(), zorder=4, label="start")
    ax_map.plot(lons_arr[-1], lats_arr[-1], marker="*", color="red", markersize=10, transform=ccrs.PlateCarree(), zorder=4, label="end")
    ax_map.set_title("13L track (dataset window)")
    ax_map.legend(loc="upper left", fontsize=8)
    cb = plt.colorbar(sc, ax=ax_map, orientation="vertical", pad=0.02, shrink=0.8)
    cb.set_label("Wind (kt)")

    # Time series
    ax_ts = fig.add_subplot(gs[0, 1])
    ax_ts.plot(times_arr, sst_list, color="tab:red", marker="o", label="SST (°C)")
    ax_ts.set_ylabel("SST (°C)", color="tab:red")
    ax_ts.tick_params(axis="y", labelcolor="tab:red")

    ax_ts2 = ax_ts.twinx()
    ax_ts2.plot(times_arr, tchp_list, color="tab:blue", marker="s", label="TCHP (kJ/cm²)")
    ax_ts2.set_ylabel("TCHP (kJ/cm²)", color="tab:blue")
    ax_ts2.tick_params(axis="y", labelcolor="tab:blue")

    ax_ts.set_xlabel("Time")
    ax_ts.set_title("Track-following SST and TCHP")
    fig.autofmt_xdate()
    fig.tight_layout()
    plt.savefig("timeseries_sst_tchp.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()

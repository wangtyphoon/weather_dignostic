import datetime as dt
import re
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd

# ==== 設定區（直接修改以下常數即可）====
# 支援多個檔名樣式，會自動展開 glob。
CSV_PATTERNS: List[str] = [
    "測站資料/466920-2025-11-*.csv",
]

# 若要直接顯示圖而非輸出檔案，將 OUTPUT_PATH 設為 None。
OUTPUT_PATH: Optional[Path] = Path("report_images/466920-all.png")

# 限制時間範圍（含邊界）；格式採 ISO，如 "2025-11-18 06:00"。
# 設為 None 代表不限制。
START_AT: Optional[str] = "2025-11-14 00:00"
END_AT: Optional[str] = "2025-11-20 23:59"
# =====================================


def read_station_csv(path: Path) -> pd.DataFrame:
    date_match = re.search(r"\d{4}-\d{2}-\d{2}", path.name)
    day = dt.datetime.strptime(date_match.group(0), "%Y-%m-%d").date() if date_match else None

    # second row has column names; skip first unit row and handle BOM with utf-8-sig
    df = pd.read_csv(path, encoding="utf-8-sig", skiprows=1)
    df = df.replace("--", pd.NA)
    df["ObsTime"] = df["ObsTime"].astype(str).str.strip().str.zfill(2)

    def obs_to_datetime(obs: str) -> dt.datetime:
        hour = int(obs)
        if hour == 24 and day:
            return dt.datetime.combine(day + dt.timedelta(days=1), dt.time(0))
        target_hour = min(hour, 23)
        if day:
            return dt.datetime.combine(day, dt.time(target_hour))
        return dt.datetime.strptime(f"{target_hour:02d}", "%H")

    df["time"] = df["ObsTime"].apply(obs_to_datetime)

    for col in ["Temperature", "StnPres", "RH"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df[["time", "Temperature", "StnPres", "RH"]].dropna(subset=["time"])


def load_all(paths: List[Path]) -> pd.DataFrame:
    frames = [read_station_csv(p) for p in paths]
    data = pd.concat(frames, ignore_index=True)
    return data.sort_values("time")


def plot_timeseries(df: pd.DataFrame, output: Optional[Path]) -> None:
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8))

    axes[0].plot(df["time"], df["Temperature"], color="tab:red")
    axes[0].set_ylabel("Temperature (\u00b0C)")

    axes[1].plot(df["time"], df["StnPres"], color="tab:blue")
    axes[1].set_ylabel("Station Pressure (hPa)")

    axes[2].plot(df["time"], df["RH"], color="tab:green")
    axes[2].set_ylabel("Relative Humidity (%)")
    axes[2].set_xlabel("Time")

    for ax in axes:
        ax.grid(True, alpha=0.3)

    fig.autofmt_xdate()
    fig.tight_layout()

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150)
        print(f"saved: {output}")
    else:
        plt.show()


def expand_csv_patterns(patterns: List[str]) -> List[Path]:
    paths: List[Path] = []
    for pattern in patterns:
        paths.extend(sorted(Path().glob(pattern)))
    unique = list(dict.fromkeys(paths))  # preserve order, remove dupes
    return unique


def to_datetime(value: Optional[str]) -> Optional[dt.datetime]:
    if value is None:
        return None
    return dt.datetime.fromisoformat(value)


def main() -> None:
    csv_paths = expand_csv_patterns(CSV_PATTERNS)
    if not csv_paths:
        raise SystemExit("no CSV files found, adjust CSV_PATTERNS in the script")

    df = load_all(sorted(csv_paths))
    start_dt = to_datetime(START_AT)
    end_dt = to_datetime(END_AT)
    if start_dt is not None:
        df = df[df["time"] >= start_dt]
    if end_dt is not None:
        df = df[df["time"] <= end_dt]

    if df.empty:
        raise SystemExit("no data loaded")
    plot_timeseries(df, OUTPUT_PATH)


if __name__ == "__main__":
    main()

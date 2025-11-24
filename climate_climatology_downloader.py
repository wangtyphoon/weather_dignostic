import cdsapi
from pathlib import Path


START_YEAR = 1979
END_YEAR = 2024
TARGET_MONTH = "11"
TARGET_DAY = "17"
TARGET_TIME = "00:00"

AREA_LIST = [80, 80, 20, 180]       # 北, 西, 南, 東
AREA_STR = "80/80/20/180"           # 北/西/南/東
PV_PARAMS = "3.128/54.128/129.128/131.128/132.128"


def _time_label(time_str: str) -> str:
    return time_str.replace(":", "")


def download_300hpa(client: cdsapi.Client, year: int) -> None:
    filename = Path(f"era5_300hPa_{year}{TARGET_MONTH}{TARGET_DAY}_{_time_label(TARGET_TIME)}.nc")
    if filename.exists():
        print(f"Skip existing file: {filename}")
        return

    client.retrieve(
        "reanalysis-era5-pressure-levels",
        {
            "product_type": "reanalysis",
            "variable": [
                "geopotential",
                "temperature",
                "u_component_of_wind",
                "v_component_of_wind",
            ],
            "pressure_level": ["300"],
            "year": str(year),
            "month": TARGET_MONTH,
            "day": TARGET_DAY,
            "time": TARGET_TIME,
            "area": AREA_LIST,
            "format": "netcdf",
        },
        str(filename),
    )


def download_pv2(client: cdsapi.Client, year: int) -> None:
    filename = Path(f"era5_pv2pv_{year}{TARGET_MONTH}{TARGET_DAY}_{_time_label(TARGET_TIME)}.nc")
    if filename.exists():
        print(f"Skip existing file: {filename}")
        return

    client.retrieve(
        "reanalysis-era5-complete",
        {
            "class": "ea",
            "date": f"{year}-{TARGET_MONTH}-{TARGET_DAY}",
            "expver": "1",
            "levtype": "pv",
            "levelist": "2000",
            "param": PV_PARAMS,
            "stream": "oper",
            "time": [TARGET_TIME],
            "type": "an",
            "grid": "0.25/0.25",
            "area": AREA_STR,
            "format": "netcdf",
        },
        str(filename),
    )

def main() -> None:
    client = cdsapi.Client()
    for year in range(START_YEAR, END_YEAR + 1):
        download_300hpa(client, year)
        download_pv2(client, year)

if __name__ == "__main__":
    main()

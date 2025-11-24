import cdsapi

c = cdsapi.Client()

c.retrieve(
    "reanalysis-era5-complete",
    {
        "class": "ea",
        "date": "2025-11-17",         # 或 "2025-11-17/to/2025-11-17"
        "expver": "1",
        "levtype": "pv",              # PV 層
        "levelist": "2000",           # 2000 * 0.001 = 2 PVU
        # 3.128 = potential temperature, 54.128 = pressure, 129.128 =                                                                                                                                                    geopotential,
        # 131.128 = u, 132.128 = v, 133.128 = q, 203.128 = relative vorticity（可視需求增減）
        "param": "3.128/54.128/129.128/131.128/132.128",
        "stream": "oper",
        'time': '00:00',
        "type": "an",
        "grid": "0.25/0.25",          # 內插到 0.25° x 0.25°
        "area": "80/80/20/180",       # 北/西/南/東（注意經度要一致：-180~180 或 0~360）
        "format": "netcdf",
    },
    "era5_pv2pv_20251117.nc")
import copernicusmarine

# 區域: 60-90W, 10-25N
min_lon, max_lon = -90.0, -60.0 
min_lat, max_lat = 10.0, 25.0

start_date = "2025-10-23"
end_date   = "2025-10-29"

# --- (A) 3D 溫度：thetao，用來算 OHC/TCHP ---
copernicusmarine.subset(
    dataset_id="cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m",
    variables=["thetao"],
    minimum_longitude=min_lon,
    maximum_longitude=max_lon,
    minimum_latitude=min_lat,
    maximum_latitude=max_lat,
    start_datetime=start_date,
    end_datetime=end_date,
    minimum_depth=0,
    maximum_depth=500,  # 為了 TCHP / OHC 抓到 500 m
    output_filename="13L_thetao_0-500m_20251023-29.nc",
)

print("3D 溫度 thetao 下載完成")

# --- (B) 2D SSH：zos ---
copernicusmarine.subset(
    dataset_id="cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
    variables=["zos"],   # 這裡不要再寫 thetao 了
    minimum_longitude=min_lon,
    maximum_longitude=max_lon,
    minimum_latitude=min_lat,
    maximum_latitude=max_lat,
    start_datetime=start_date,
    end_datetime=end_date,
    # 2D 場不需要 depth，可以不寫或保留 0
    output_filename="13L_zos_20251023-29.nc",
)

print("SSH zos 下載完成")

# --- (C) 3D 流場：uo, vo（選擇性） ---
copernicusmarine.subset(
    dataset_id="cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m",
    variables=["uo", "vo"],
    minimum_longitude=min_lon,
    maximum_longitude=max_lon,
    minimum_latitude=min_lat,
    maximum_latitude=max_lat,
    start_datetime=start_date,
    end_datetime=end_date,
    minimum_depth=0,
    maximum_depth=500,
    output_filename="13L_cur_0-500m_20251023-29.nc",
)

print("3D 流場 uo/vo 下載完成")

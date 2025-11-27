import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import cm

# ===================== 使用者設定 =====================
YEAR = 2025
DATE_STR = f"{YEAR}-11-17"          # 11/17
time_str = DATE_STR + "T00:00"

# 資料與氣候平均設定
DATA_DIR = "era5"
CLIM_START_YEAR = 1979
CLIM_END_YEAR = 2024
g = 9.80665  # ERA5 geopotential 單位轉換用

# 作圖範圍（可調整）
# 這裡設定：緯度 20–80N, 經度 80E–180E（與下載範圍一致）
lat_min, lat_max = 20, 80
lon_west, lon_east = 80, 180

# 「藍色框」大致範圍（你可以微調）
# box_lat_min, box_lat_max = 30, 75
# box_lon_west, box_lon_east = 100, 160   # 約 100E–160E 在 180 座標下
# ===================================================

PVU_CMAP = cm.get_cmap("bwr").copy()
# PVU_CMAP.set_under("#dae4f5")
# PVU_CMAP.set_over("#f5dada")


def normalize_coords(ds: xr.Dataset) -> xr.Dataset:
    """Rename common ERA5 coordinate names to a consistent set."""
    rename_map = {}
    if 'valid_time' in ds.dims or 'valid_time' in ds.coords:
        rename_map['valid_time'] = 'time'
    if 'pressure_level' in ds.dims or 'pressure_level' in ds.coords:
        rename_map['pressure_level'] = 'level'
    if 'latitude' in ds.dims or 'latitude' in ds.coords:
        rename_map['latitude'] = 'lat'
    if 'longitude' in ds.dims or 'longitude' in ds.coords:
        rename_map['longitude'] = 'lon'
    if 'isobaricInhPa' in ds.dims or 'isobaricInhPa' in ds.coords:
        rename_map['isobaricInhPa'] = 'level'
    return ds.rename(rename_map)


def select_level(da: xr.DataArray, target_level: int) -> xr.DataArray:
    """Safely pick a pressure level if it exists, otherwise return as-is."""
    if 'level' in da.dims:
        return da.sel(level=target_level)
    return da


def load_variable(ds: xr.Dataset, candidates) -> xr.DataArray:
    """Return the first existing variable from candidates."""
    for cand in candidates:
        if cand in ds:
            return ds[cand]
    raise KeyError(f"找不到候選變數：{candidates}")


def pick_existing_file(prefix: str, year: int) -> str:
    """Pick file with/without _0000 suffix in DATA_DIR."""
    candidates = [
        os.path.join(DATA_DIR, f"{prefix}_{year}1117.nc"),
        os.path.join(DATA_DIR, f"{prefix}_{year}1117_0000.nc"),
    ]
    for fp in candidates:
        if os.path.exists(fp):
            return fp
    raise FileNotFoundError(f"{prefix} {year} 的檔案不存在，請確認路徑。")


def assign_lon_and_slice(da: xr.DataArray) -> xr.DataArray:
    """Convert lon to 0-360 and crop to requested domain."""
    da = da.assign_coords(lon=(da.lon % 360))
    return da.sel(lat=slice(lat_max, lat_min), lon=slice(lon_west, lon_east))


def load_climatology(prefix: str, var_candidates, level: int | None = None) -> xr.DataArray:
    """
    讀取 1979-2024 的 ERA5 資料並計算平均。
    使用 combine_attrs='drop_conflicts' 避免 history 屬性衝突。
    """
    files = []
    for y in range(CLIM_START_YEAR, CLIM_END_YEAR + 1):
        try:
            files.append(pick_existing_file(prefix, y))
        except FileNotFoundError:
            continue
    if not files:
        raise FileNotFoundError(f"找不到 {prefix} 的 climatology 檔案")

    arrays = []
    for fp in files:
        ds = normalize_coords(xr.open_dataset(fp))
        da = select_level(load_variable(ds, var_candidates), level)
        arrays.append(da)

    clim_all = xr.concat(arrays, dim="time", combine_attrs="drop_conflicts")
    return clim_all.mean(dim="time")


def load_data():
    # 300 hPa 資料
    z_file = pick_existing_file("era5_300hPa", YEAR)
    ds300 = normalize_coords(xr.open_dataset(z_file))
    Z300 = select_level(ds300['z'] / g, 300)
    U300 = select_level(ds300['u'], 300)
    V300 = select_level(ds300['v'], 300)

    # 2 PVU 位溫
    for prefix in ["era5_pv2pv", "era5_theta2pvu"]:
        try:
            theta_file = pick_existing_file(prefix, YEAR)
            break
        except FileNotFoundError:
            continue
    else:
        raise FileNotFoundError("找不到 2PVU 位溫檔案，請確認檔名或路徑。")

    ds_th = normalize_coords(xr.open_dataset(theta_file))
    THETA2 = load_variable(ds_th, ['pt', 'potential_temperature', 'theta'])

    # 選定時間切片（這裡以 00UTC 為例）
    Z300_t   = Z300.sel(time=time_str).squeeze(drop=True)
    U300_t   = U300.sel(time=time_str).squeeze(drop=True)
    V300_t   = V300.sel(time=time_str).squeeze(drop=True)
    THETA2_t = THETA2.sel(time=time_str).squeeze(drop=True)

    # 裁切空間範圍
    Z300_t   = assign_lon_and_slice(Z300_t)
    U300_t   = assign_lon_and_slice(U300_t)
    V300_t   = assign_lon_and_slice(V300_t)
    THETA2_t = assign_lon_and_slice(THETA2_t)

    # 以資料實際經緯度做軸範圍，避免超出下載範圍
    lon_min_val = float(Z300_t.lon.min())
    lon_max_val = float(Z300_t.lon.max())
    lat_min_val = float(Z300_t.lat.min())
    lat_max_val = float(Z300_t.lat.max())

    return (Z300_t, U300_t, V300_t, THETA2_t,
            lon_min_val, lon_max_val, lat_min_val, lat_max_val)


def compute_anomaly(field: xr.DataArray, climatology: xr.DataArray) -> xr.DataArray:
    """field - climatology，兩者需在相同網格。"""
    return field - climatology


# def add_box(ax):
#     """畫出藍色框，範圍以上面 box_* 參數為準"""
#     import cartopy.crs as ccrs
#     from matplotlib.patches import Rectangle

#     rect = Rectangle(
#         (box_lon_west, box_lat_min),
#         box_lon_east - box_lon_west,
#         box_lat_max - box_lat_min,
#         linewidth=1.5, edgecolor='blue', facecolor='none',
#         transform=ccrs.PlateCarree()
#     )
#     ax.add_patch(rect)


def main():
    (Z300_t, U300_t, V300_t, THETA2_t,
     lon_min_val, lon_max_val, lat_min_val, lat_max_val) = load_data()

    lon = Z300_t.lon
    lat = Z300_t.lat

    # 讀取 1979-2024 氣候平均並計算 anomaly
    Z300_clim = assign_lon_and_slice(
        load_climatology("era5_300hPa", ['z'], level=300) / g
    )
    THETA2_clim = assign_lon_and_slice(
        load_climatology("era5_pv2pv", ['pt', 'potential_temperature', 'theta'])
    )

    Z300_anom   = compute_anomaly(Z300_t, Z300_clim)
    THETA2_anom = compute_anomaly(THETA2_t, THETA2_clim)

    proj = ccrs.PlateCarree()

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 5),
        subplot_kw={'projection': proj},
        constrained_layout=True
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    # 共用繪圖設定
    for ax in axes.ravel():
        ax.set_extent([lon_min_val, lon_max_val, lat_min_val, lat_max_val],
                      crs=proj)
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 8}
        gl.ylabel_style = {'size': 8}

    # ---------- (a) Z300 與 Z300' ----------
    cs_a = axes[0].contourf(
        lon, lat, Z300_anom,
        levels=np.arange(-800, 801, 100),
        cmap='bwr', extend='both',
        transform=proj
    )
    z300_clim_min = float(np.floor(np.nanmin(Z300_clim) / 80) * 80)
    z300_clim_max = float(np.ceil(np.nanmax(Z300_clim) / 80) * 80)
    c_a = axes[0].contour(
        lon, lat, Z300_clim,
        levels=np.arange(z300_clim_min, z300_clim_max + 1e-6, 80),
        colors='k', linewidths=1,
        transform=proj
    )
    axes[0].clabel(c_a, fmt='%d', fontsize=6)
    # add_box(axes[0])
    axes[0].set_title('(a) Z300 and Z300\' anomaly')

    # ---------- (b) θ2PVU 與 θ2PVU' ----------
    print(np.max(THETA2_anom), np.min(THETA2_anom))
    cs_b = axes[1].contourf(
        lon, lat, THETA2_anom,
        levels=np.arange(-30, 30, 3),
        cmap=PVU_CMAP, extend='both',
        transform=proj
    )
    theta_clim_min = float(np.floor(np.nanmin(THETA2_clim) / 4) * 4)
    theta_clim_max = float(np.ceil(np.nanmax(THETA2_clim) / 4) * 4)
    c_b = axes[1].contour(
        lon, lat, THETA2_clim,
        levels=np.arange(theta_clim_min, theta_clim_max + 1e-6, 5),
        colors='k', linewidths=1,
        transform=proj
    )
    axes[1].clabel(c_b, fmt='%d', fontsize=6)
    # add_box(axes[1])
    axes[1].set_title('(b) θ2PVU and θ2PVU\' anomaly')

    # 加上 colorbar
    fig.colorbar(cs_a, ax=axes[0], orientation='horizontal', pad=0.05)
    fig.colorbar(cs_b, ax=axes[1], orientation='horizontal', pad=0.05)

    plt.suptitle(f"ERA5 {DATE_STR} Cold Surge Panels", fontsize=14)
    plt.show()


if __name__ == "__main__":
    main()

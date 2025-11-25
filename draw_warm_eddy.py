import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

# file names already in /mnt/d/weather_dignostic
theta = xr.open_dataset("13L_thetao_0-500m_20251023-29.nc")["thetao"]
cur   = xr.open_dataset("13L_cur_0-500m_20251023-29.nc")
zos   = xr.open_dataset("13L_zos_20251023-29.nc")["zos"]

rho = 1025.0       # kg/m3
cp  = 3994.0       # J/(kg·K)
tidx = 6           # 0..6 for 2025-10-23..29

# layer thickness from staggered depth midpoints
depth_vals = theta.depth.values
edges = np.zeros(depth_vals.size + 1)
edges[1:-1] = (depth_vals[:-1] + depth_vals[1:]) / 2
edges[0] = depth_vals[0] - (depth_vals[1] - depth_vals[0]) / 2
edges[-1] = depth_vals[-1] + (depth_vals[-1] - depth_vals[-2]) / 2
thick = np.diff(edges)
thick_da = xr.DataArray(thick, coords={"depth": theta.depth}, dims=["depth"])

# TCHP relative to 26 °C converted to kJ/cm^2 (1 J/m^2 = 1e-7 kJ/cm^2)
excess_warm = (theta - 26).where(theta > 26, 0)
tchp_kjcm2 = (excess_warm * thick_da).sum("depth") * rho * cp * 1e-7

# SST and surface currents (use shallowest model level)
sst = theta.isel(time=tidx).sel(depth=theta.depth.min(), method="nearest")
uo_sfc = cur["uo"].isel(time=tidx).sel(depth=cur.depth.min(), method="nearest")
vo_sfc = cur["vo"].isel(time=tidx).sel(depth=cur.depth.min(), method="nearest")

lon, lat = theta.longitude, theta.latitude
lon2d, lat2d = np.meshgrid(lon, lat)
extent = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]
proj = ccrs.PlateCarree()

# --- kinematic diagnostics (relative vorticity & Okubo-Weiss) ---
R = 6371000.0
lat_rad = np.deg2rad(lat.values)
lon_rad = np.deg2rad(lon.values)
coslat = np.cos(lat_rad)

du_dy = np.gradient(uo_sfc.values, lat_rad, axis=0) / R
du_dx = np.gradient(uo_sfc.values, lon_rad, axis=1) / (R * coslat[:, None])
dv_dy = np.gradient(vo_sfc.values, lat_rad, axis=0) / R
dv_dx = np.gradient(vo_sfc.values, lon_rad, axis=1) / (R * coslat[:, None])

zeta = dv_dx - du_dy
s1 = du_dx - dv_dy  # normal strain
s2 = dv_dx + du_dy  # shear strain
ow = s1**2 + s2**2 - zeta**2

def get_center_at_time(best_track_path, target_dt64):
    """
    Return (lat, lon, label) matching target_dt64 to the nearest hour.
    Supports rows with full date+time or time-only (date carried forward).
    """
    if not os.path.exists(best_track_path):
        return None

    target_hr = target_dt64.astype("datetime64[h]")
    current_date = None
    try:
        with open(best_track_path, "r", encoding="utf-8") as f:
            lines = f.readlines()[2:]  # skip headers
    except Exception:
        return None

    for line in lines:
        if not line.strip():
            continue
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        dt = None
        lat_idx = None
        # date + time (e.g., 2025-10-23 00:00:00)
        if len(parts) >= 8 and "-" in parts[1] and ":" in parts[2]:
            date_str, time_str = parts[1], parts[2]
            dt = np.datetime64(f"{date_str}T{time_str}")
            current_date = dt.astype("datetime64[D]")
            lat_idx = 4
        # time only, use current_date (e.g., 06:00:00)
        elif ":" in parts[1] and current_date is not None:
            time_str = parts[1]
            dt = np.datetime64(f"{str(current_date)}T{time_str}")
            lat_idx = 3

        if dt is None or lat_idx is None:
            continue

        if dt.astype("datetime64[h]") == target_hr:
            try:
                lat_val = float(parts[lat_idx])
                lon_val = float(parts[lat_idx + 1])
            except (ValueError, IndexError):
                return None
            label = dt.astype("datetime64[m]").astype(str)
            return lat_val, lon_val, label
    return None

center_pt = get_center_at_time("13L_best_track.txt", theta.time.values[tidx])
print("Center point:", center_pt)
def cartopy_map(field, title, cmap, units, zos_field=None, contour_levels=None, center=None):
    fig = plt.figure(figsize=(9, 6))
    ax = plt.axes(projection=proj)
    mesh = ax.pcolormesh(lon, lat, field, shading="auto", cmap=cmap, transform=proj)
    if zos_field is not None:
        ax.streamplot(
            lon2d, lat2d, uo_sfc.values, vo_sfc.values,
            color="k", linewidth=0.7, density=1.3, arrowsize=0.9, transform=proj,
        )
    if center is not None:
        lat_c, lon_c, label = center
        ax.plot(lon_c, lat_c, marker="*", color="red", markersize=10, transform=proj, zorder=5)
        ax.text(lon_c + 0.2, lat_c + 0.2, label, color="red", fontsize=8, transform=proj, zorder=5)
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.6, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    ax.set_extent(extent, crs=proj)
    cb = plt.colorbar(mesh, ax=ax, orientation="vertical", pad=0.02, shrink=0.85)
    cb.set_label(units)
    ax.set_title(title)
    plt.tight_layout()

# Plot TCHP (kJ/cm^2) with SSH contours
cartopy_map(
    tchp_kjcm2.isel(time=tidx),
    f"TCHP (ref 26°C) 0–500 m {str(tchp_kjcm2.time.values[tidx])[:10]}",
    "turbo",
    "kJ/cm²",
    zos_field=zos.isel(time=tidx),
    center=center_pt,
)

# Plot SST
cartopy_map(sst, f"SST {str(sst.time.values)[:10]}", "turbo", "°C", center=center_pt)

# Plot surface currents with SSH background
fig = plt.figure(figsize=(9, 6))
ax = plt.axes(projection=proj)
bg = ax.pcolormesh(lon, lat, zos.isel(time=tidx), shading="auto", cmap="RdBu_r", transform=proj)
cb = plt.colorbar(bg, ax=ax, orientation="vertical", pad=0.02, shrink=0.85)
cb.set_label("SSH (m)")
ax.streamplot(
    lon2d, lat2d, uo_sfc.values, vo_sfc.values,
    color="k", linewidth=0.8, density=1.3, arrowsize=1.1, transform=proj
)
if center_pt is not None:
    lat_c, lon_c, label_c = center_pt
    ax.plot(lon_c, lat_c, marker="*", color="red", markersize=10, transform=proj, zorder=5)
    ax.text(lon_c + 0.2, lat_c + 0.2, label_c, color="red", fontsize=8, transform=proj, zorder=5)
ax.add_feature(cfeature.LAND, facecolor="lightgray")
ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
ax.add_feature(cfeature.BORDERS, linewidth=0.4)
gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.6, linestyle="--")
gl.top_labels = False
gl.right_labels = False
ax.set_extent(extent, crs=proj)
ax.set_title(f"Surface Currents {str(uo_sfc.time.values)[:10]}")
plt.tight_layout()

# Plot relative vorticity (1/s)
fig = plt.figure(figsize=(9, 6))
ax = plt.axes(projection=proj)
mesh = ax.pcolormesh(lon, lat, zeta, shading="auto", cmap="RdBu_r", transform=proj)
cb = plt.colorbar(mesh, ax=ax, orientation="vertical", pad=0.02, shrink=0.85)
cb.set_label("Relative vorticity (1/s)")
ax.streamplot(lon2d, lat2d, uo_sfc.values, vo_sfc.values, color="k", linewidth=0.5, density=1.1, transform=proj)
if center_pt is not None:
    lat_c, lon_c, label_c = center_pt
    ax.plot(lon_c, lat_c, marker="*", color="red", markersize=10, transform=proj, zorder=5)
    ax.text(lon_c + 0.2, lat_c + 0.2, label_c, color="red", fontsize=8, transform=proj, zorder=5)
ax.add_feature(cfeature.LAND, facecolor="lightgray")
ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
ax.add_feature(cfeature.BORDERS, linewidth=0.4)
gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.6, linestyle="--")
gl.top_labels = False
gl.right_labels = False
ax.set_extent(extent, crs=proj)
ax.set_title(f"Surface Relative Vorticity {str(uo_sfc.time.values)[:10]}")
plt.tight_layout()

# Plot Okubo-Weiss parameter
fig = plt.figure(figsize=(9, 6))
ax = plt.axes(projection=proj)
mesh = ax.pcolormesh(lon, lat, ow, shading="auto", cmap="PuOr_r", transform=proj)
cb = plt.colorbar(mesh, ax=ax, orientation="vertical", pad=0.02, shrink=0.85)
cb.set_label("Okubo-Weiss (1/s²)")
ax.streamplot(lon2d, lat2d, uo_sfc.values, vo_sfc.values, color="k", linewidth=0.5, density=1.1, transform=proj)
if center_pt is not None:
    lat_c, lon_c, label_c = center_pt
    ax.plot(lon_c, lat_c, marker="*", color="red", markersize=10, transform=proj, zorder=5)
    ax.text(lon_c + 0.2, lat_c + 0.2, label_c, color="red", fontsize=8, transform=proj, zorder=5)
ax.add_feature(cfeature.LAND, facecolor="lightgray")
ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
ax.add_feature(cfeature.BORDERS, linewidth=0.4)
gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.6, linestyle="--")
gl.top_labels = False
gl.right_labels = False
ax.set_extent(extent, crs=proj)
ax.set_title(f"Okubo-Weiss {str(uo_sfc.time.values)[:10]}")
plt.tight_layout()

plt.show()

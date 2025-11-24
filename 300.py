import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-pressure-levels',
    {
        'product_type': 'reanalysis',
        'variable': [
            'geopotential', 'temperature',
            'u_component_of_wind', 'v_component_of_wind',
        ],
        'pressure_level': ['300'],
        'year': '2025',
        'month': '11',
        'day': '17',
        'time': '00:00',
        'area': [80, 80, 20, 180],  # 北, 西, 南, 東
        'format': 'netcdf',
    },
    'era5_300hPa_20251117.nc'
)
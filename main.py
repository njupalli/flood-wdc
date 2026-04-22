import rasterio
import numpy as np
import geopandas as gpd
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.features import rasterize
from rasterio.windows import from_bounds
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import pandas as pd
import richdem as rd
import joblib
import requests
import os
from datetime import datetime

# ===============================
# 1️⃣ LOAD DEM
# ===============================
dem_path = "USGS_13_n26w081_20251216.tif"

with rasterio.open(dem_path) as dem_src:
    dem = dem_src.read(1)
    dem_transform = dem_src.transform
    dem_crs = dem_src.crs
    dem_mask = dem_src.read_masks(1)
    dem = np.where(dem_mask == 0, np.nan, dem)

print("DEM shape:", dem.shape)

# ===============================
# 2️⃣ LOAD & ALIGN RAINFALL
# ===============================
rainfall_path = "nws_precip_1day_20250915_conus.tif"

with rasterio.open(rainfall_path) as rf_src:
    rainfall = rf_src.read(1)
    rf_transform = rf_src.transform
    rf_crs = rf_src.crs
    rf_mask = rf_src.read_masks(1)
    rainfall = np.where(rf_mask == 0, np.nan, rainfall)

rainfall_aligned = np.empty_like(dem, dtype=np.float32)

reproject(
    source=rainfall,
    destination=rainfall_aligned,
    src_transform=rf_transform,
    src_crs=rf_crs,
    dst_transform=dem_transform,
    dst_crs=dem_crs,
    resampling=Resampling.bilinear
)

print("Rainfall aligned.")

# ===============================
# 3️⃣ LOAD FLOOD SHAPEFILE
# ===============================
flood_shp_path = "12011C_20251212/S_FLD_HAZ_AR.shp"

flood = gpd.read_file(flood_shp_path)
flood = flood.to_crs(dem_crs)

flood["label"] = flood["FLD_ZONE"].apply(
    lambda x: 1 if str(x) in ["AE", "VE", "A", "AH", "AO"] else 0
)

shapes = ((geom, value) for geom, value in zip(flood.geometry, flood.label))

flood_raster = rasterize(
    shapes=shapes,
    out_shape=dem.shape,
    transform=dem_transform,
    fill=0,
    dtype="uint8"
)

print("Flood labels created.")

# ===============================
# 4️⃣ LOAD & ALIGN IMPERVIOUS
# ===============================
impervious_path = "Annual_NLCD_FctImp_2024_CU_C1V1/Annual_NLCD_FctImp_2024_CU_C1V1.tif"

with rasterio.open(impervious_path) as imp_src:
    left, bottom, right, top = rasterio.transform.array_bounds(
        dem.shape[0], dem.shape[1], dem_transform
    )
    left_t, bottom_t, right_t, top_t = transform_bounds(
        dem_crs, imp_src.crs, left, bottom, right, top
    )
    window = from_bounds(left_t, bottom_t, right_t, top_t, imp_src.transform)
    impervious = imp_src.read(1, window=window)
    impervious_aligned = np.empty_like(dem, dtype=np.float32)

    reproject(
        source=impervious,
        destination=impervious_aligned,
        src_transform=imp_src.window_transform(window),
        src_crs=imp_src.crs,
        dst_transform=dem_transform,
        dst_crs=dem_crs,
        resampling=Resampling.bilinear
    )

print("Impervious aligned.")

# ===============================
# 5️⃣ COMPUTE SLOPE
# ===============================
xres = dem_transform[0]
yres = -dem_transform[4]
dz_dy, dz_dx = np.gradient(dem, yres, xres)
slope = np.degrees(np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2)))

print("Slope calculated.")

# ===============================
# 6️⃣ COMPUTE FLOW ACCUMULATION
# ===============================
rd_dem = rd.rdarray(dem, no_data=np.nan)
rd_dem.geotransform = dem_transform
rd.FillDepressions(rd_dem, in_place=True)
flow_accum = rd.FlowAccumulation(rd_dem, method='D8')
flow_accum = np.array(flow_accum)
flow_accum = np.log1p(flow_accum)

print("Flow accumulation calculated.")

# ===============================
# 7️⃣ SAMPLING & TRAINING
# ===============================
dem_f = dem.flatten()
slope_f = slope.flatten()
flow_f = flow_accum.flatten()
rain_f = rainfall_aligned.flatten()
imp_f = impervious_aligned.flatten()
y_f = flood_raster.flatten()

valid_mask = (~np.isnan(dem_f) & ~np.isnan(slope_f) & ~np.isnan(flow_f) & ~np.isnan(rain_f) & ~np.isnan(imp_f))
valid_indices = np.where(valid_mask)[0]
y_valid = y_f[valid_indices]

flood_idx = valid_indices[y_valid == 1]
nonflood_idx = valid_indices[y_valid == 0]
n_samples = min(len(flood_idx), 200000)

np.random.seed(42)
flood_sample = np.random.choice(flood_idx, n_samples, replace=False)
nonflood_sample = np.random.choice(nonflood_idx, n_samples, replace=False)
sample_idx = np.concatenate([flood_sample, nonflood_sample])

X_sample = np.column_stack(
    (dem_f[sample_idx], slope_f[sample_idx], flow_f[sample_idx], rain_f[sample_idx], imp_f[sample_idx])).astype(
    np.float32)
y_sample = y_f[sample_idx]

X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
joblib.dump(rf, "flood_model.pkl")

# ===============================
# 8️⃣ GENERATE STATIC PROBABILITY MAP
# ===============================
probability_map = np.full(dem.shape, np.nan, dtype=np.float32)
rows_dim, cols_dim = dem.shape
tile_size = 512

for r in range(0, rows_dim, tile_size):
    for c in range(0, cols_dim, tile_size):
        re, ce = min(r + tile_size, rows_dim), min(c + tile_size, cols_dim)
        d_t, s_t, f_t, r_t, i_t = dem[r:re, c:ce], slope[r:re, c:ce], flow_accum[r:re, c:ce], rainfall_aligned[r:re,
                                                                                              c:ce], impervious_aligned[
                                                                                                     r:re, c:ce]
        m = (~np.isnan(d_t) & ~np.isnan(s_t) & ~np.isnan(f_t) & ~np.isnan(r_t) & ~np.isnan(i_t))
        if np.any(m):
            X_tile = np.column_stack((d_t[m], s_t[m], f_t[m], r_t[m], i_t[m])).astype(np.float32)
            probs = rf.predict_proba(X_tile)[:, 1]
            t_prob = np.full(d_t.shape, np.nan, dtype=np.float32)
            t_prob[m] = probs
            probability_map[r:re, c:ce] = t_prob

# ===============================
# 9️⃣ SAVE DATA FOR TABLEAU (WITH FEATURES)
# ===============================
rows, cols = np.where((~np.isnan(probability_map)) & (probability_map != -9999.0))
r_f, c_f = rows[::100], cols[::100]
xs, ys = rasterio.transform.xy(dem_transform, r_f, c_f)

df_presentation = pd.DataFrame({
    "longitude": xs,
    "latitude": ys,
    "elevation": dem[r_f, c_f],
    "slope": slope[r_f, c_f],
    "flow": flow_accum[r_f, c_f],
    "rainfall": rainfall_aligned[r_f, c_f],
    "impervious": impervious_aligned[r_f, c_f],
    "flood_prob": probability_map[r_f, c_f]
})
df_presentation.to_csv("florida_flood_FINAL.csv", index=False)
print("Emergency file created with all features!")

# ===============================
# 🔟 API TOOL: LIVE WEATHER DATA
# ===============================
API_KEY = "2e526f09c28fbc8a7699bb45329847cb"
LAT, LON = 26.1224, -80.1373
base_df = pd.read_csv("florida_flood_FINAL.csv")


def get_flood_predictions(rain_mm, label):
    # Ensure columns match the training features exactly
    X = base_df[['elevation', 'slope', 'flow', 'rainfall', 'impervious']].values
    X[:, 3] = rain_mm  # Update rainfall column
    probs = rf.predict_proba(X)[:, 1]

    temp_df = base_df[['longitude', 'latitude']].copy()
    temp_df['flood_prob'] = probs
    temp_df['time_label'] = label
    temp_df['timestamp'] = datetime.now()
    return temp_df


# Fetch Current
curr_url = f"https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"
curr_data = requests.get(curr_url).json()
current_rain = curr_data.get('rain', {}).get('1h', 0)

# Fetch Forecast
fore_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"
fore_data = requests.get(fore_url).json()

timeline_results = []
print(f"Processing Current Rain: {current_rain}mm")
timeline_results.append(get_flood_predictions(current_rain, "Current"))

for entry in fore_data['list'][:3]:
    time_str = entry['dt_txt']
    rain_f = entry.get('rain', {}).get('3h', 0) / 3
    print(f"Processing Forecast {time_str}: {rain_f}mm")
    timeline_results.append(get_flood_predictions(rain_f, f"Forecast: {time_str}"))

final_tool_df = pd.concat(timeline_results)
final_tool_df.to_csv("FLORIDA_FLOOD_TOOL_DATA.csv", index=False)
print("Tool Data Ready! Use FLORIDA_FLOOD_TOOL_DATA.csv in Tableau.")

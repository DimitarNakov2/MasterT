import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

from pathlib import Path

# ============================================================
# 1. SETTINGS
# ============================================================

PROJECT_DIR = Path('/shared/home/dimitar.nakov/MasterT/MastersThesis')
RAW_DIR = PROJECT_DIR / 'data' / 'raw_data'
OUTPUT_DIR = PROJECT_DIR / 'data' / 'processed'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHL_PATH = RAW_DIR / 'cmems_obs-oc_med_bgc-plankton_my_l3-multi-1km_P1D_CHL_11.54E-22.79E_35.45N-45.99N_2016-06-01-2026-06-01.nc'
BBP_PATH = RAW_DIR / 'cmems_obs-oc_med_bgc-optics_my_l3-multi-1km_P1D_BBP443_11.54E-22.79E_35.45N-45.99N_2016-06-01-2026-06-01.nc'
SST_PATH = RAW_DIR / 'SST_MED_SST_L3S_NRT_OBSERVATIONS_010_012_b_multi-vars_11.55E-22.79E_35.45N-46.00N_2016-06-01-2026-06-01.nc'

CHL_VAR = 'CHL'
BBP_VAR = 'BBP443'
SST_VAR = 'adjusted_sea_surface_temperature'
SST_QUALITY_VAR = 'quality_level'

SST_MIN_QUALITY = 4

CHL_MIN = 0.01
CHL_MAX = 100.0
BBP_MIN = 1e-8
SST_MIN_C = -5.0
SST_MAX_C = 40.0

PATCH_SIZE = 128
PATCH_STRIDE = 128
MIN_OCEAN_FRACTION = 0.20
MIN_TARGET_VALID_FRACTION = 0.05
MAX_TEMPORAL_OFFSET = 3

TRAIN_START = '2016-06-01'
TRAIN_END = '2024-12-31'
VAL_START = '2025-01-01'
VAL_END = '2025-12-31'
TEST_START = '2026-01-01'
TEST_END = '2026-06-01'

print('CHL:', CHL_PATH)
print('BBP:', BBP_PATH)
print('SST:', SST_PATH)


# ============================================================
# 2. LOAD DATASETS
# ============================================================

ds_chl = xr.open_dataset(CHL_PATH, chunks={'time': 16})
ds_bbp = xr.open_dataset(BBP_PATH, chunks={'time': 16})
ds_sst = xr.open_dataset(SST_PATH, chunks={'time': 8})

chl = ds_chl[CHL_VAR]
bbp = ds_bbp[BBP_VAR]
sst_raw = ds_sst[SST_VAR]
sst_quality_raw = ds_sst[SST_QUALITY_VAR]

print('CHL variable:', CHL_VAR)
print('BBP variable:', BBP_VAR)
print('SST variable:', SST_VAR)
print('SST quality variable:', SST_QUALITY_VAR)
print('CHL shape:', chl.shape)
print('BBP shape:', bbp.shape)
print('SST shape:', sst_raw.shape)
print('SST units:', sst_raw.attrs.get('units'))


# ============================================================
# 3. ALIGN BBP443 AND HIGH-RESOLUTION SST TO CHL
# ============================================================

bbp = bbp.reindex(time=chl.time)
same_bbp_lat = np.array_equal(bbp.latitude.values, chl.latitude.values)
same_bbp_lon = np.array_equal(bbp.longitude.values, chl.longitude.values)
if not (same_bbp_lat and same_bbp_lon):
    print('Regridding BBP443 to CHL grid...')
    bbp = bbp.interp(latitude=chl.latitude, longitude=chl.longitude, method='linear')

sst_raw = sst_raw.reindex(time=chl.time)
sst_quality_raw = sst_quality_raw.reindex(time=chl.time)

sst = sst_raw - 273.15
sst.attrs['units'] = 'degree_Celsius'

# Build the quality mask on the native high-resolution SST grid.
sst_valid_hr = (
    np.isfinite(sst)
    & (sst_quality_raw >= SST_MIN_QUALITY)
    & (sst >= SST_MIN_C)
    & (sst <= SST_MAX_C)
)
sst_good_hr = sst.where(sst_valid_hr)

print('Regridding high-resolution SST to CHL grid...')
sst = sst_good_hr.interp(
    latitude=chl.latitude,
    longitude=chl.longitude,
    method='linear',
)

# Quality level is categorical: nearest-neighbour only.
sst_quality = sst_quality_raw.interp(
    latitude=chl.latitude,
    longitude=chl.longitude,
    method='nearest',
)

sst_valid = (
    np.isfinite(sst)
    & (sst_quality >= SST_MIN_QUALITY)
    & (sst >= SST_MIN_C)
    & (sst <= SST_MAX_C)
)

chl_valid = np.isfinite(chl) & (chl >= CHL_MIN) & (chl <= CHL_MAX)
bbp_valid = np.isfinite(bbp) & (bbp > BBP_MIN)

print('Aligned shapes:')
print(' CHL:', chl.shape)
print(' BBP:', bbp.shape)
print(' SST:', sst.shape)
print(' SST quality:', sst_quality.shape)


# ============================================================
# 4. MASKS + AUXILIARY AVAILABILITY
# ============================================================

# chl_valid, bbp_valid and sst_valid were already created in the
# alignment cell. In particular, sst_valid includes quality_level >= 4.

# Ocean = pixel that contains at least one valid CHL measurement over the period.
ocean_mask = chl_valid.any(dim='time').compute()

print('Ocean pixels:', int(ocean_mask.sum()), '/', ocean_mask.size)

# How useful are the auxiliary variables specifically where CHL is missing?
chl_missing_ocean = (~chl_valid) & ocean_mask

missing_count = chl_missing_ocean.sum().compute().item()
bbp_when_chl_missing = (bbp_valid & chl_missing_ocean).sum().compute().item()
sst_when_chl_missing = (sst_valid & chl_missing_ocean).sum().compute().item()

bbp_when_chl_missing_pct = (
    100.0 * bbp_when_chl_missing / missing_count if missing_count else np.nan
)
sst_when_chl_missing_pct = (
    100.0 * sst_when_chl_missing / missing_count if missing_count else np.nan
)

print(f'CHL missing ocean pixels: {missing_count:,}')
print(f'BBP443 available where CHL missing: {bbp_when_chl_missing_pct:.3f}%')
print(f'SST available where CHL missing: {sst_when_chl_missing_pct:.3f}%')


# ============================================================
# 5. TRAIN / VALIDATION / TEST SPLITS
# ============================================================

dates = pd.to_datetime(chl.time.values)

def indices_between(start, end):
    return np.where((dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end)))[0]

train_indices = indices_between(TRAIN_START, TRAIN_END)
val_indices = indices_between(VAL_START, VAL_END)
test_indices = indices_between(TEST_START, TEST_END)

print("Train days:", len(train_indices), dates[train_indices[0]], "->", dates[train_indices[-1]])
print("Val days:", len(val_indices), dates[val_indices[0]], "->", dates[val_indices[-1]])
print("Test days:", len(test_indices), dates[test_indices[0]], "->", dates[test_indices[-1]])

# ============================================================
# 6. TRAIN-ONLY TRANSFORMS / NORMALIZATION
# ============================================================

# CHL and BBP are positive and strongly skewed -> natural log.
chl_log = xr.where(chl_valid, np.log(chl), np.nan)
bbp_log = xr.where(bbp_valid, np.log(bbp), np.nan)

# SST remains linear in degrees Celsius.
train_chl_log = chl_log.sel(time=slice(TRAIN_START, TRAIN_END))
train_bbp_log = bbp_log.sel(time=slice(TRAIN_START, TRAIN_END))
train_sst = sst.sel(time=slice(TRAIN_START, TRAIN_END))

def mean_std(da):
    mean = float(da.mean(skipna=True).compute().item())
    std = float(da.std(skipna=True).compute().item())
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError(f"Invalid normalization statistics: mean={mean}, std={std}")
    return mean, std

chl_mean, chl_std = mean_std(train_chl_log)
bbp_mean, bbp_std = mean_std(train_bbp_log)
sst_mean, sst_std = mean_std(train_sst)

print(f"CHL log: mean={chl_mean:.6f}, std={chl_std:.6f}")
print(f"BBP443 log: mean={bbp_mean:.6f}, std={bbp_std:.6f}")
print(f"SST C: mean={sst_mean:.6f}, std={sst_std:.6f}")

chl_norm = ((chl_log - chl_mean) / chl_std).astype("float32")
bbp_norm = ((bbp_log - bbp_mean) / bbp_std).astype("float32")
sst_norm = ((sst - sst_mean) / sst_std).astype("float32")

# ============================================================
# 7. TEMPORAL FALLBACK ANALYSIS FOR CHL (same logic as before)
# ============================================================

# This computes masks only; it does not load all CHL values into RAM.
mask_np = chl_valid.compute().values.astype(bool)
ocean_np = ocean_mask.values.astype(bool)

rows = []
for i in range(MAX_TEMPORAL_OFFSET, len(dates) - MAX_TEMPORAL_OFFSET):
    target_missing = (~mask_np[i]) & ocean_np
    n_missing = int(target_missing.sum())
    if n_missing == 0:
        continue

    visible_1 = mask_np[i - 1] | mask_np[i + 1]
    visible_2 = visible_1 | mask_np[i - 2] | mask_np[i + 2]
    visible_3 = visible_2 | mask_np[i - 3] | mask_np[i + 3]

    rows.append({
        "date": dates[i],
        "visible_pm1_pct": 100.0 * (target_missing & visible_1).sum() / n_missing,
        "visible_pm2_pct": 100.0 * (target_missing & visible_2).sum() / n_missing,
        "visible_pm3_pct": 100.0 * (target_missing & visible_3).sum() / n_missing,
    })

fallback_df = pd.DataFrame(rows)

# ============================================================
# 8. PATCH INDEX
# ============================================================

def sliding_starts(size, patch_size, stride):
    starts = list(range(0, max(size - patch_size + 1, 1), stride))
    last = max(size - patch_size, 0)
    if not starts or starts[-1] != last:
        starts.append(last)
    return sorted(set(starts))

def build_patch_index(ocean_mask_da, patch_size, min_ocean_fraction, stride=None):
    if stride is None:
        stride = patch_size

    arr = ocean_mask_da.values.astype(bool)
    h, w = arr.shape
    rows = []

    for y in sliding_starts(h, patch_size, stride):
        for x in sliding_starts(w, patch_size, stride):
            patch = arr[y:y + patch_size, x:x + patch_size]
            if patch.shape != (patch_size, patch_size):
                continue
            frac = float(patch.mean())
            if frac >= min_ocean_fraction:
                rows.append({"y": y, "x": x, "ocean_fraction": frac})

    return pd.DataFrame(rows)

patch_index = build_patch_index(
    ocean_mask, PATCH_SIZE, MIN_OCEAN_FRACTION, PATCH_STRIDE
)

print("Usable patches:", len(patch_index))
print(patch_index["ocean_fraction"].describe())


# ============================================================
# 9. SAMPLE INDEX
# ============================================================

def build_sample_index(split_name, indices, max_offset=3):
    rows = []
    split_set = set(int(i) for i in indices)

    for idx in indices:
        idx = int(idx)
        required = list(range(idx - max_offset, idx + max_offset + 1))

        if required[0] < 0 or required[-1] >= len(dates):
            continue
        if not all(j in split_set for j in required):
            continue

        actual = dates[idx - max_offset: idx + max_offset + 1]
        expected = pd.date_range(
            dates[idx] - pd.Timedelta(days=max_offset),
            dates[idx] + pd.Timedelta(days=max_offset),
            freq="D",
        )
        if not np.array_equal(actual.values, expected.values):
            continue

        for _, patch in patch_index.iterrows():
            y = int(patch.y)
            x = int(patch.x)

            ocean_patch = ocean_np[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
            target_valid_patch = mask_np[idx, y:y + PATCH_SIZE, x:x + PATCH_SIZE]

            ocean_pixels = int(ocean_patch.sum())
            if ocean_pixels == 0:
                continue

            valid_target_pixels = int((target_valid_patch & ocean_patch).sum())
            target_valid_fraction = valid_target_pixels / ocean_pixels

            if target_valid_fraction < MIN_TARGET_VALID_FRACTION:
                continue

            rows.append({
                "split": split_name,
                "time_idx": idx,
                "date": dates[idx],
                "y": y,
                "x": x,
                "ocean_fraction": float(patch.ocean_fraction),
                "target_valid_pixels": valid_target_pixels,
                "target_valid_fraction": float(target_valid_fraction),
            })

    return rows

sample_rows = []
sample_rows += build_sample_index("train", train_indices, MAX_TEMPORAL_OFFSET)
sample_rows += build_sample_index("validation", val_indices, MAX_TEMPORAL_OFFSET)
sample_rows += build_sample_index("test", test_indices, MAX_TEMPORAL_OFFSET)

sample_index = pd.DataFrame(sample_rows)

print("Samples per split:")
print(sample_index.groupby("split").size())
print("\nUnique target days:")
print(sample_index.groupby("split")["date"].nunique())
print("\nTarget valid fraction:")


# ============================================================
# 10. VISUAL SANITY CHECK
# ============================================================

# Pick one date and show all three aligned inputs.
idx = int(train_indices[len(train_indices) // 2])

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

chl.isel(time=idx).plot(ax=axes[0], robust=True)
axes[0].set_title(f"CHL - {dates[idx].date()}")

bbp.isel(time=idx).plot(ax=axes[1], robust=True)
axes[1].set_title("BBP443 aligned to CHL grid")

sst.isel(time=idx).plot(ax=axes[2], robust=True)
axes[2].set_title("SST (°C) aligned to CHL grid")

plt.tight_layout()
plt.show()

# ============================================================
# 11. SAVE PREPROCESSING METADATA
# ============================================================

# As in the original CHL-only preprocessing notebook, we do not save a
# second copy of the full gridded data. The models will read the original
# NetCDF files and use the statistics/configuration saved here.

sample_index.to_csv(OUTPUT_DIR / "sample_index.csv", index=False)
patch_index.to_csv(OUTPUT_DIR / "patch_index.csv", index=False)
ocean_mask.to_netcdf(OUTPUT_DIR / "ocean_mask.nc")
fallback_df.to_csv(OUTPUT_DIR / "temporal_fallback_analysis.csv", index=False)

normalization_df = pd.DataFrame([
    {"variable": "CHL", "transform": "natural_log", "train_mean": chl_mean, "train_std": chl_std},
    {"variable": "BBP443", "transform": "natural_log", "train_mean": bbp_mean, "train_std": bbp_std},
    {"variable": "SST", "transform": "linear_celsius", "train_mean": sst_mean, "train_std": sst_std},
])
normalization_df.to_csv(OUTPUT_DIR / "normalization.csv", index=False)

config = {
    "data_version": "chl_sst_bbp443_v1",
    # Keep raw NetCDF paths, matching the original CHL preprocessing approach.
    "data_path": str(CHL_PATH),  # backward-compatible alias for existing CHL loaders
    "chl_path": str(CHL_PATH),
    "bbp443_path": str(BBP_PATH),
    "sst_path": str(SST_PATH),
    "chl_variable": CHL_VAR,
    "bbp443_variable": BBP_VAR,
    "sst_variable": SST_VAR,
    "sst_quality_variable": SST_QUALITY_VAR,
    "sst_min_quality": int(SST_MIN_QUALITY),
    "chl_min": CHL_MIN,
    "chl_max": CHL_MAX,
    "bbp_min": BBP_MIN,
    "sst_min_c": SST_MIN_C,
    "sst_max_c": SST_MAX_C,
    "patch_size": PATCH_SIZE,
    "patch_stride": PATCH_STRIDE,
    "min_ocean_fraction": MIN_OCEAN_FRACTION,
    "min_target_valid_fraction": MIN_TARGET_VALID_FRACTION,
    "max_temporal_offset": MAX_TEMPORAL_OFFSET,
    "train_start": TRAIN_START,
    "train_end": TRAIN_END,
    "validation_start": VAL_START,
    "validation_end": VAL_END,
    "test_start": TEST_START,
    "test_end": TEST_END,
    "chl_transform": "natural_log",
    "chl_train_mean": float(chl_mean),
    "chl_train_std": float(chl_std),
    # Backward-compatible aliases used by current CHL-only loaders/evaluators.
    "transform": "natural_log",
    "normalization": "zscore_using_training_data",
    "train_mean": float(chl_mean),
    "train_std": float(chl_std),
    "bbp443_transform": "natural_log",
    "bbp443_train_mean": float(bbp_mean),
    "bbp443_train_std": float(bbp_std),
    "sst_transform": "linear_celsius",
    "sst_train_mean": float(sst_mean),
    "sst_train_std": float(sst_std),
    "bbp443_available_where_chl_missing_pct": float(bbp_when_chl_missing_pct),
    "sst_available_where_chl_missing_pct": float(sst_when_chl_missing_pct),
    # Tell future loaders that auxiliaries must be aligned to the CHL grid.
    "aux_alignment_reference": "CHL",
    "aux_time_alignment": "reindex_to_chl_time",
    "aux_spatial_alignment": "linear_interp_to_chl_lat_lon_if_needed",
}

with open(OUTPUT_DIR / "preprocessing_config.json", "w") as f:
    json.dump(config, f, indent=4)

print("\nSaved preprocessing files:")
for p in sorted(OUTPUT_DIR.iterdir()):
    print(" -", p.name)


# ============================================================
# 12. FINAL VALIDATION
# ============================================================

assert chl.shape == bbp.shape == sst.shape == sst_quality.shape
assert np.array_equal(chl.time.values, bbp.time.values)
assert np.array_equal(chl.time.values, sst.time.values)
assert np.array_equal(chl.latitude.values, bbp.latitude.values)
assert np.array_equal(chl.latitude.values, sst.latitude.values)
assert np.array_equal(chl.longitude.values, bbp.longitude.values)
assert np.array_equal(chl.longitude.values, sst.longitude.values)

required_files = [
    'sample_index.csv',
    'patch_index.csv',
    'normalization.csv',
    'ocean_mask.nc',
    'temporal_fallback_analysis.csv',
    'preprocessing_config.json',
]

for name in required_files:
    path = OUTPUT_DIR / name
    assert path.exists(), f'Missing preprocessing output: {path}'

print('\nPreprocessing completed successfully.')
print('High-resolution SST was quality-filtered and aligned to the CHL grid.')
print('No Zarr/model-data copy was created.')
print('Models should read the NetCDF files listed in preprocessing_config.json.')
print('\nSamples:')
print(sample_index.groupby('split').size())

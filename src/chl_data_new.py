"""
PyTorch Dataset for CMEMS Mediterranean chlorophyll-a reconstruction.

Supported modes
---------------
SPATIAL:
- Loads only the target day plus the real mask day.
- Returns 3 channels:
    0 current masked CHL
    1 current visible mask
    2 ocean mask

TEMPORAL:
- Loads the t-3 ... t+3 window and builds the existing fallback context.
- Returns 11 channels:
    0      past filled CHL
    1..3   past source one-hot (offset 1, 2, 3)
    4      artificially masked current CHL
    5      current visible-observation mask
    6      future filled CHL
    7..9   future source one-hot (offset 1, 2, 3)
    10     ocean mask

DIRECT:
- Loads the same t-3 ... t+3 window used by the temporal U-Net.
- Builds DIRECT's prioritized temporal fallback context:
    previous: t-1 -> t-2 -> t-3
    future:   t+1 -> t+2 -> t+3
- Preserves the temporal origin of every filled pixel.
- The centre frame uses the same artificial real-cloud mask as all other
  models, preserving fair evaluation.
- Returns:
    direct_context : [3,H,W]
        [past_filled, current_masked, future_filled]

    past_source : [H,W] integer source map
        0 unavailable, 1 from t-1, 2 from t-2, 3 from t-3

    future_source : [H,W] integer source map
        0 unavailable, 1 from t+1, 2 from t+2, 3 from t+3

    direct_temporal_masks : [8,H,W]
        0..2 previous source one-hot in DIRECT order:
             [t-3, t-2, t-1]
        3    centre observed
        4    centre missing-or-land
        5..7 future source one-hot in DIRECT order:
             [t+3, t+2, t+1]

- FUSE rectification and construction of the noisy centre field are performed
  in train_direct.py / evaluate_direct.py, not in the dataset.

MAMBA:
- Loads a longer centered temporal sequence (default 15 days).
- Keeps natural missing Chl-a values as zero and provides an explicit
  observation-validity mask for every day.
- Can optionally include SST and/or BBP443 using aux_mode:
    none        -> CHL only (3 channels/day; backward compatible)
    sst         -> CHL + SST (5 channels/day)
    bbp443      -> CHL + BBP443 (5 channels/day)
    sst_bbp443  -> CHL + SST + BBP443 (7 channels/day)
- Per-variable value channels are normalized with TRAINING-ONLY statistics.
- Every auxiliary value channel has its own validity-mask channel.
- Target-day BBP443 is masked with the same artificial cloud mask as CHL,
  because both are ocean-colour products and should not leak through a
  simulated cloud. SST is NOT artificially cloud-masked because it is an
  independent auxiliary observation and is allowed to remain available.
- The centre day is replaced by the exact same artificially masked centre
  observation used by the other benchmark models.
- Context is leakage-safe: frames outside the current chronological split
  are zeroed instead of reading across train/validation/test boundaries.
- Returns:
    mamba_sequence       : [T,3,H,W]
    mamba_chl            : [T,1,H,W]
    mamba_observation_mask: [T,1,H,W]
    mamba_ocean_mask     : [T,1,H,W]
    mamba_time_valid     : [T] (1 if the date is available within the split)
    mamba_offsets        : [T] integer offsets around the target date
    mamba_doy            : [T,2] seasonal sine/cosine encoding

All modes share exactly the same:
- sample_index.csv
- chronological train/validation/test split
- preprocessing_config.json
- log transform + training-set normalization
- ocean mask
- real same-split cloud-mask self-supervision
- deterministic validation/test masking
- target/loss masks

Therefore all supported models can be compared on the same held-out pixels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, get_worker_info
import xarray as xr


class ChlorophyllReconstructionDataset(Dataset):
    def __init__(
        self,
        processed_dir: str | Path,
        split: str = "train",
        input_mode: str = "temporal",
        sample_index_file: str = "sample_index.csv",
        config_file: str = "preprocessing_config.json",
        ocean_mask_file: str = "ocean_mask.nc",
        seed: int = 42,
        max_mask_attempts: int = 20,
        min_hidden_pixels: int = 64,
        mamba_sequence_length: int = 15,
        aux_mode: str = "none",
    ):
        """
        Parameters
        ----------
        processed_dir:
            Directory containing the files saved by data_prep.ipynb.
        split:
            One of {"train", "validation", "test"}.
        input_mode:
            "spatial"  -> 3-channel spatial U-Net input.
            "temporal" -> existing 11-channel temporal U-Net input.
            "direct"   -> DIRECT fallback context + temporal-origin masks.
            "mamba"    -> long temporal sequence [T,3,H,W].
        mamba_sequence_length:
            Odd number of daily frames used by Mamba. Default 15 gives
            7 days before + target day + 7 days after.
        aux_mode:
            Auxiliary variables to expose/use. One of:
            "none", "sst", "bbp443", "sst_bbp443".
            Default "none" preserves exact compatibility with existing models.
        seed:
            Base seed. Validation/test mask selection is deterministic.
        max_mask_attempts:
            Number of real masks to try when looking for enough artificially
            hidden ground-truth pixels.
        min_hidden_pixels:
            Desired minimum number of artificially hidden known pixels in
            a patch. If no sampled mask reaches this value, the best sampled
            mask is used. This does NOT fabricate ground truth.
        """
        super().__init__()

        if split not in {"train", "validation", "test"}:
            raise ValueError(
                f"split must be 'train', 'validation', or 'test', got {split!r}"
            )

        self.processed_dir = Path(processed_dir)
        self.split = split
        self.seed = int(seed)
        self.max_mask_attempts = int(max_mask_attempts)
        self.min_hidden_pixels = int(min_hidden_pixels)
        self.input_mode = str(input_mode).lower()
        self.mamba_sequence_length = int(mamba_sequence_length)
        self.aux_mode = str(aux_mode).lower()

        if self.aux_mode not in {"none", "sst", "bbp443", "sst_bbp443"}:
            raise ValueError(
                "aux_mode must be one of 'none', 'sst', 'bbp443', "
                f"'sst_bbp443'; got {self.aux_mode!r}"
            )

        self.use_sst = self.aux_mode in {"sst", "sst_bbp443"}
        self.use_bbp443 = self.aux_mode in {"bbp443", "sst_bbp443"}

        if self.input_mode not in {
            "spatial",
            "temporal",
            "direct",
            "mamba",
        }:
            raise ValueError(
                "input_mode must be 'spatial', 'temporal', 'direct', "
                f"or 'mamba', got {self.input_mode!r}"
            )

        if (
            self.mamba_sequence_length < 1
            or self.mamba_sequence_length % 2 == 0
        ):
            raise ValueError(
                "mamba_sequence_length must be a positive odd integer, "
                f"got {self.mamba_sequence_length}"
            )

        self.mamba_half_window = (
            self.mamba_sequence_length // 2
        )

        # ------------------------------------------------------------
        # Load saved preprocessing configuration
        # ------------------------------------------------------------
        config_path = self.processed_dir / config_file
        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.data_path = str(self.config["data_path"])
        self.chl_variable = self.config.get("chl_variable", "CHL")

        self.chl_min = float(self.config["chl_min"])
        self.chl_max = float(self.config["chl_max"])

        self.patch_size = int(self.config["patch_size"])
        self.max_offset = int(self.config.get("max_temporal_offset", 3))

        self.train_mean = float(self.config["train_mean"])
        self.train_std = float(self.config["train_std"])

        if self.train_std <= 0:
            raise ValueError("train_std must be > 0")

        # ------------------------------------------------------------
        # Auxiliary-variable configuration
        # ------------------------------------------------------------
        self.sst_path = self.config.get("sst_path")
        self.sst_variable = self.config.get(
            "sst_variable", "adjusted_sea_surface_temperature"
        )
        self.sst_quality_variable = self.config.get(
            "sst_quality_variable", "quality_level"
        )
        self.sst_min_quality = float(
            self.config.get("sst_min_quality", 4)
        )
        self.sst_min_c = float(self.config.get("sst_min_c", -5.0))
        self.sst_max_c = float(self.config.get("sst_max_c", 40.0))
        self.sst_train_mean = float(
            self.config.get("sst_train_mean", 0.0)
        )
        self.sst_train_std = float(
            self.config.get("sst_train_std", 1.0)
        )

        self.bbp443_path = self.config.get("bbp443_path")
        self.bbp443_variable = self.config.get(
            "bbp443_variable", "BBP443"
        )
        self.bbp_min = float(self.config.get("bbp_min", 1e-8))
        self.bbp443_train_mean = float(
            self.config.get("bbp443_train_mean", 0.0)
        )
        self.bbp443_train_std = float(
            self.config.get("bbp443_train_std", 1.0)
        )

        if self.use_sst:
            if not self.sst_path:
                raise ValueError(
                    "aux_mode requests SST but preprocessing_config.json "
                    "does not contain sst_path."
                )
            if self.sst_train_std <= 0:
                raise ValueError("sst_train_std must be > 0")

        if self.use_bbp443:
            if not self.bbp443_path:
                raise ValueError(
                    "aux_mode requests BBP443 but preprocessing_config.json "
                    "does not contain bbp443_path."
                )
            if self.bbp443_train_std <= 0:
                raise ValueError("bbp443_train_std must be > 0")

        # ------------------------------------------------------------
        # Load sample metadata
        # ------------------------------------------------------------
        index_path = self.processed_dir / sample_index_file
        sample_index = pd.read_csv(index_path, parse_dates=["date"])

        self.samples = (
            sample_index[sample_index["split"] == split]
            .reset_index(drop=True)
        )

        if len(self.samples) == 0:
            raise ValueError(f"No samples found for split={split!r}")

        # Chronological split boundaries. Mamba uses these to avoid reading
        # temporal context across train/validation/test boundaries.
        self.split_min_time_idx = int(
            self.samples["time_idx"].min()
        )
        self.split_max_time_idx = int(
            self.samples["time_idx"].max()
        )

        # Unique valid target times in this split. These are also used
        # as a leakage-safe pool of real masks.
        self.mask_time_indices = np.sort(
            self.samples["time_idx"].astype(int).unique()
        )

        # ------------------------------------------------------------
        # Load fixed ocean mask into memory
        # ------------------------------------------------------------
        ocean_path = self.processed_dir / ocean_mask_file

        try:
            ocean_da = xr.open_dataarray(ocean_path)
            self.ocean_mask = ocean_da.load().values.astype(bool)
            ocean_da.close()
        except Exception:
            ocean_ds = xr.open_dataset(ocean_path)
            first_var = list(ocean_ds.data_vars)[0]
            self.ocean_mask = ocean_ds[first_var].load().values.astype(bool)
            ocean_ds.close()

        # Opened lazily per process/worker.
        self._ds: Optional[xr.Dataset] = None
        self._chl: Optional[xr.DataArray] = None
        self._spatial_dims: Optional[list[str]] = None

        self._sst_ds: Optional[xr.Dataset] = None
        self._sst: Optional[xr.DataArray] = None
        self._sst_quality: Optional[xr.DataArray] = None

        self._bbp_ds: Optional[xr.Dataset] = None
        self._bbp: Optional[xr.DataArray] = None

        self._rng: Optional[np.random.Generator] = None

    # ------------------------------------------------------------------
    # Multiprocessing safety
    # ------------------------------------------------------------------
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_ds"] = None
        state["_chl"] = None
        state["_spatial_dims"] = None
        state["_sst_ds"] = None
        state["_sst"] = None
        state["_sst_quality"] = None
        state["_bbp_ds"] = None
        state["_bbp"] = None
        state["_rng"] = None
        return state

    def _ensure_open(self):
        if self._ds is not None:
            return

        self._ds = xr.open_dataset(
            self.data_path,
            chunks=None,
            cache=False,
        )

        self._chl = self._ds[self.chl_variable]

        if "time" not in self._chl.dims:
            raise ValueError("CHL variable must contain a 'time' dimension")

        self._spatial_dims = [
            d for d in self._chl.dims if d != "time"
        ]

        if len(self._spatial_dims) != 2:
            raise ValueError(
                "Expected exactly two spatial CHL dimensions; "
                f"found {self._spatial_dims}"
            )

        if self.use_sst:
            self._sst_ds = xr.open_dataset(
                self.sst_path,
                chunks=None,
                cache=False,
            )
            self._sst = self._sst_ds[self.sst_variable]
            self._sst_quality = self._sst_ds[
                self.sst_quality_variable
            ]

        if self.use_bbp443:
            self._bbp_ds = xr.open_dataset(
                self.bbp443_path,
                chunks=None,
                cache=False,
            )
            self._bbp = self._bbp_ds[self.bbp443_variable]

    def _get_rng(self) -> np.random.Generator:
        if self._rng is None:
            worker = get_worker_info()
            worker_id = 0 if worker is None else worker.id
            self._rng = np.random.default_rng(
                self.seed + 1009 * worker_id
            )
        return self._rng

    # ------------------------------------------------------------------
    # Data loading / preprocessing
    # ------------------------------------------------------------------
    def _patch_slices(self, y: int, x: int):
        return {
            self._spatial_dims[0]: slice(
                y, y + self.patch_size
            ),
            self._spatial_dims[1]: slice(
                x, x + self.patch_size
            ),
        }

    def _clean_and_normalize(
        self,
        arr: np.ndarray,
        ocean_patch: np.ndarray,
    ):
        """
        Return
        ------
        normalized:
            float32; invalid/missing pixels = 0.
        valid:
            bool mask of valid observed Chl-a pixels.
        """
        arr = np.asarray(arr, dtype=np.float32)

        valid = (
            np.isfinite(arr)
            & (arr >= self.chl_min)
            & (arr <= self.chl_max)
            & ocean_patch
        )

        normalized = np.zeros_like(
            arr,
            dtype=np.float32,
        )

        if valid.any():
            logged = np.log(arr[valid])
            normalized[valid] = (
                logged - self.train_mean
            ) / self.train_std

        return normalized, valid


    # ------------------------------------------------------------------
    # Auxiliary-variable loading / normalization
    # ------------------------------------------------------------------
    def _chl_patch_coords(self, y: int, x: int):
        """
        Return target CHL latitude/longitude coordinates for one patch.
        """
        self._ensure_open()

        lat_dim, lon_dim = self._spatial_dims

        # The preprocessing pipeline uses latitude/longitude. Keep a fallback
        # to the spatial dimension names so the loader remains robust.
        lat_name = "latitude" if "latitude" in self._chl.coords else lat_dim
        lon_name = "longitude" if "longitude" in self._chl.coords else lon_dim

        lat = self._chl[lat_name].isel(
            {lat_dim: slice(y, y + self.patch_size)}
        ).values
        lon = self._chl[lon_name].isel(
            {lon_dim: slice(x, x + self.patch_size)}
        ).values

        return (
            np.asarray(lat, dtype=np.float64),
            np.asarray(lon, dtype=np.float64),
        )

    @staticmethod
    def _coord_slice(coord: xr.DataArray, lo: float, hi: float):
        """
        Build a slice that respects ascending or descending coordinates.
        """
        first = float(coord.values[0])
        last = float(coord.values[-1])
        if first <= last:
            return slice(lo, hi)
        return slice(hi, lo)

    def _time_index_for_date(
        self,
        da: xr.DataArray,
        date_value,
    ) -> Optional[int]:
        """
        Find an auxiliary dataset time index matching one CHL date exactly
        at day resolution. Returns None if unavailable.
        """
        target_day = np.datetime64(
            pd.Timestamp(date_value).normalize(),
            "D",
        )

        times = da["time"].values.astype("datetime64[D]")
        pos = int(np.searchsorted(times, target_day))
        if pos < len(times) and times[pos] == target_day:
            return pos
        return None

    def _clean_and_normalize_bbp(
        self,
        arr: np.ndarray,
        ocean_patch: np.ndarray,
    ):
        arr = np.asarray(arr, dtype=np.float32)

        valid = (
            np.isfinite(arr)
            & (arr > self.bbp_min)
            & ocean_patch
        )

        normalized = np.zeros_like(arr, dtype=np.float32)

        if valid.any():
            logged = np.log(arr[valid])
            normalized[valid] = (
                logged - self.bbp443_train_mean
            ) / self.bbp443_train_std

        return normalized, valid

    def _clean_and_normalize_sst_c(
        self,
        arr_c: np.ndarray,
        quality: np.ndarray,
        ocean_patch: np.ndarray,
    ):
        arr_c = np.asarray(arr_c, dtype=np.float32)
        quality = np.asarray(quality, dtype=np.float32)

        valid = (
            np.isfinite(arr_c)
            & np.isfinite(quality)
            & (quality >= self.sst_min_quality)
            & (arr_c >= self.sst_min_c)
            & (arr_c <= self.sst_max_c)
            & ocean_patch
        )

        normalized = np.zeros_like(arr_c, dtype=np.float32)

        if valid.any():
            normalized[valid] = (
                arr_c[valid] - self.sst_train_mean
            ) / self.sst_train_std

        return normalized, valid

    def _load_bbp_patch(
        self,
        time_idx: int,
        y: int,
        x: int,
    ):
        """
        Load one BBP443 patch on the CHL target grid.

        BBP443 normally uses exactly the same grid as CHL. If coordinates differ,
        the selected local field is linearly interpolated to the CHL patch grid.
        """
        self._ensure_open()

        ocean_patch = self.ocean_mask[
            y:y + self.patch_size,
            x:x + self.patch_size,
        ]

        date_value = self._ds["time"].isel(time=int(time_idx)).values
        aux_idx = self._time_index_for_date(self._bbp, date_value)

        if aux_idx is None:
            zeros = np.zeros(
                (self.patch_size, self.patch_size),
                dtype=np.float32,
            )
            return zeros, zeros.astype(bool)

        target_lat, target_lon = self._chl_patch_coords(y, x)

        same_lat = (
            self._bbp.sizes.get("latitude") == self._chl.sizes.get("latitude")
            and np.array_equal(
                self._bbp["latitude"].values,
                self._chl["latitude"].values,
            )
        )
        same_lon = (
            self._bbp.sizes.get("longitude") == self._chl.sizes.get("longitude")
            and np.array_equal(
                self._bbp["longitude"].values,
                self._chl["longitude"].values,
            )
        )

        if same_lat and same_lon:
            raw = self._bbp.isel(
                time=aux_idx,
                latitude=slice(y, y + self.patch_size),
                longitude=slice(x, x + self.patch_size),
            ).load().values.astype(np.float32)
        else:
            # Local interpolation only; avoids loading/regridding the whole file.
            lat_step = float(np.median(np.abs(np.diff(self._bbp["latitude"].values))))
            lon_step = float(np.median(np.abs(np.diff(self._bbp["longitude"].values))))
            lat_sel = self._coord_slice(
                self._bbp["latitude"],
                float(target_lat.min() - 2 * lat_step),
                float(target_lat.max() + 2 * lat_step),
            )
            lon_sel = self._coord_slice(
                self._bbp["longitude"],
                float(target_lon.min() - 2 * lon_step),
                float(target_lon.max() + 2 * lon_step),
            )

            local = self._bbp.isel(time=aux_idx).sel(
                latitude=lat_sel,
                longitude=lon_sel,
            )

            raw = local.interp(
                latitude=xr.DataArray(target_lat, dims="latitude"),
                longitude=xr.DataArray(target_lon, dims="longitude"),
                method="linear",
            ).load().values.astype(np.float32)

        return self._clean_and_normalize_bbp(
            raw,
            ocean_patch,
        )

    def _load_sst_patch(
        self,
        time_idx: int,
        y: int,
        x: int,
    ):
        """
        Load one high-resolution SST patch and interpolate it locally onto the
        CHL patch grid.

        Quality filtering is applied on the native SST grid BEFORE interpolation.
        `quality_level` itself is nearest-neighbour interpolated because it is
        categorical. This reproduces the preprocessing notebook logic without
        writing a second gridded SST file.
        """
        self._ensure_open()

        ocean_patch = self.ocean_mask[
            y:y + self.patch_size,
            x:x + self.patch_size,
        ]

        date_value = self._ds["time"].isel(time=int(time_idx)).values
        aux_idx = self._time_index_for_date(self._sst, date_value)

        if aux_idx is None:
            zeros = np.zeros(
                (self.patch_size, self.patch_size),
                dtype=np.float32,
            )
            return zeros, zeros.astype(bool)

        target_lat, target_lon = self._chl_patch_coords(y, x)

        lat_step = float(
            np.median(np.abs(np.diff(self._sst["latitude"].values)))
        )
        lon_step = float(
            np.median(np.abs(np.diff(self._sst["longitude"].values)))
        )

        lat_sel = self._coord_slice(
            self._sst["latitude"],
            float(target_lat.min() - 2 * lat_step),
            float(target_lat.max() + 2 * lat_step),
        )
        lon_sel = self._coord_slice(
            self._sst["longitude"],
            float(target_lon.min() - 2 * lon_step),
            float(target_lon.max() + 2 * lon_step),
        )

        native_sst = self._sst.isel(time=aux_idx).sel(
            latitude=lat_sel,
            longitude=lon_sel,
        )
        native_quality = self._sst_quality.isel(time=aux_idx).sel(
            latitude=lat_sel,
            longitude=lon_sel,
        )

        # Original SST units are Kelvin.
        native_sst_c = native_sst - 273.15

        native_valid = (
            np.isfinite(native_sst_c)
            & np.isfinite(native_quality)
            & (native_quality >= self.sst_min_quality)
            & (native_sst_c >= self.sst_min_c)
            & (native_sst_c <= self.sst_max_c)
        )

        native_good = native_sst_c.where(native_valid)

        target_lat_da = xr.DataArray(
            target_lat,
            dims="latitude",
        )
        target_lon_da = xr.DataArray(
            target_lon,
            dims="longitude",
        )

        sst_interp = native_good.interp(
            latitude=target_lat_da,
            longitude=target_lon_da,
            method="linear",
        ).load().values.astype(np.float32)

        quality_interp = native_quality.interp(
            latitude=target_lat_da,
            longitude=target_lon_da,
            method="nearest",
        ).load().values.astype(np.float32)

        return self._clean_and_normalize_sst_c(
            sst_interp,
            quality_interp,
            ocean_patch,
        )

    def _load_aux_current_patch(
        self,
        time_idx: int,
        y: int,
        x: int,
    ):
        """
        Load selected auxiliary fields for one target day.
        Returns a dict with normalized values and validity masks.
        """
        out = {}

        if self.use_sst:
            sst, sst_valid = self._load_sst_patch(
                time_idx,
                y,
                x,
            )
            out["sst"] = sst
            out["sst_valid"] = sst_valid

        if self.use_bbp443:
            bbp, bbp_valid = self._load_bbp_patch(
                time_idx,
                y,
                x,
            )
            out["bbp443"] = bbp
            out["bbp443_valid"] = bbp_valid

        return out

    def _load_aux_mamba_sequence(
        self,
        center_idx: int,
        y: int,
        x: int,
        time_valid: np.ndarray,
        offsets: np.ndarray,
    ):
        """
        Load selected auxiliary variables for the same leakage-safe Mamba
        sequence positions as CHL.

        Returns dict entries with shape [T,H,W].
        """
        T = len(offsets)
        H = W = self.patch_size
        out = {}

        if self.use_sst:
            out["sst"] = np.zeros((T, H, W), dtype=np.float32)
            out["sst_valid"] = np.zeros((T, H, W), dtype=bool)

        if self.use_bbp443:
            out["bbp443"] = np.zeros((T, H, W), dtype=np.float32)
            out["bbp443_valid"] = np.zeros((T, H, W), dtype=bool)

        for pos, offset in enumerate(offsets):
            if not bool(time_valid[pos]):
                continue

            tidx = int(center_idx + int(offset))

            if self.use_sst:
                val, valid = self._load_sst_patch(tidx, y, x)
                out["sst"][pos] = val
                out["sst_valid"][pos] = valid

            if self.use_bbp443:
                val, valid = self._load_bbp_patch(tidx, y, x)
                out["bbp443"][pos] = val
                out["bbp443_valid"][pos] = valid

        return out

    def _load_target_patch(
        self,
        time_idx: int,
        y: int,
        x: int,
    ):
        """
        Load only the target-day CHL patch.
        """
        self._ensure_open()

        sl = self._patch_slices(y, x)

        raw = self._chl.isel(
            time=int(time_idx),
            **sl,
        ).load().values.astype(np.float32)

        ocean_patch = self.ocean_mask[
            y:y + self.patch_size,
            x:x + self.patch_size,
        ]

        target, target_valid = (
            self._clean_and_normalize(
                raw,
                ocean_patch,
            )
        )

        return (
            target,
            target_valid,
            ocean_patch,
        )

    def _load_window_patch(
        self,
        center_idx: int,
        y: int,
        x: int,
    ):
        """
        Load [t-max_offset, ..., t+max_offset] in one NetCDF read.
        Used by the existing temporal U-Net mode.
        """
        self._ensure_open()

        sl = self._patch_slices(y, x)

        start = center_idx - self.max_offset
        stop = center_idx + self.max_offset + 1

        da = self._chl.isel(
            time=slice(start, stop),
            **sl,
        )

        raw = da.load().values.astype(np.float32)

        expected = 2 * self.max_offset + 1
        if raw.shape[0] != expected:
            raise IndexError(
                f"Temporal sample at time_idx={center_idx} "
                f"does not have a complete {expected}-frame window."
            )

        ocean_patch = self.ocean_mask[
            y:y + self.patch_size,
            x:x + self.patch_size,
        ]

        normalized = np.zeros_like(
            raw,
            dtype=np.float32,
        )
        valid = np.zeros_like(
            raw,
            dtype=bool,
        )

        for k in range(raw.shape[0]):
            normalized[k], valid[k] = (
                self._clean_and_normalize(
                    raw[k],
                    ocean_patch,
                )
            )

        return (
            normalized,
            valid,
            ocean_patch,
        )

    def _load_direct_window_patch(
        self,
        center_idx: int,
        y: int,
        x: int,
    ):
        """
        Legacy helper for loading exactly [t-1, t, t+1].

        The corrected DIRECT mode does NOT use this helper. DIRECT uses the
        t±1 -> t±2 -> t±3 prioritized fallback described in the thesis.
        """
        self._ensure_open()

        if center_idx <= 0:
            raise IndexError(
                f"DIRECT sample time_idx={center_idx} has no t-1 frame."
            )

        n_times = int(self._chl.sizes["time"])
        if center_idx >= n_times - 1:
            raise IndexError(
                f"DIRECT sample time_idx={center_idx} has no t+1 frame."
            )

        sl = self._patch_slices(y, x)

        da = self._chl.isel(
            time=slice(
                center_idx - 1,
                center_idx + 2,
            ),
            **sl,
        )

        raw = da.load().values.astype(np.float32)

        if raw.shape[0] != 3:
            raise IndexError(
                f"Expected 3 DIRECT frames at time_idx={center_idx}, "
                f"got shape {raw.shape}"
            )

        ocean_patch = self.ocean_mask[
            y:y + self.patch_size,
            x:x + self.patch_size,
        ]

        normalized = np.zeros_like(
            raw,
            dtype=np.float32,
        )
        valid = np.zeros_like(
            raw,
            dtype=bool,
        )

        for k in range(3):
            normalized[k], valid[k] = (
                self._clean_and_normalize(
                    raw[k],
                    ocean_patch,
                )
            )

        return (
            normalized,
            valid,
            ocean_patch,
        )

    def _load_mamba_sequence_patch(
        self,
        center_idx: int,
        y: int,
        x: int,
    ):
        """
        Load a leakage-safe centered Mamba sequence.

        Returns
        -------
        sequence_values:
            [T,H,W] normalized Chl-a; unavailable/missing pixels = 0.
        sequence_valid:
            [T,H,W] boolean observation mask.
        time_valid:
            [T] boolean flag indicating that the requested day exists inside
            the current chronological split and raw dataset.
        offsets:
            [T] integer temporal offsets relative to the target day.
        sequence_doy:
            [T,2] sine/cosine day-of-year encodings.
        ocean_patch:
            [H,W] boolean ocean mask.

        Notes
        -----
        We deliberately do not borrow frames from another split. Near a split
        boundary, unavailable context positions stay zero with time_valid=0.
        This preserves the original benchmark samples without temporal leakage.
        """
        self._ensure_open()

        T = self.mamba_sequence_length
        half = self.mamba_half_window

        offsets = np.arange(
            -half,
            half + 1,
            dtype=np.int64,
        )

        requested = (
            int(center_idx)
            + offsets
        )

        n_time = int(
            self._chl.sizes["time"]
        )

        time_valid = (
            (requested >= 0)
            & (requested < n_time)
            & (
                requested
                >= self.split_min_time_idx
            )
            & (
                requested
                <= self.split_max_time_idx
            )
        )

        ocean_patch = self.ocean_mask[
            y:y + self.patch_size,
            x:x + self.patch_size,
        ]

        values = np.zeros(
            (
                T,
                self.patch_size,
                self.patch_size,
            ),
            dtype=np.float32,
        )

        valid = np.zeros(
            (
                T,
                self.patch_size,
                self.patch_size,
            ),
            dtype=bool,
        )

        sequence_doy = np.zeros(
            (T, 2),
            dtype=np.float32,
        )

        sl = self._patch_slices(y, x)

        valid_positions = np.flatnonzero(
            time_valid
        )

        if len(valid_positions) > 0:
            valid_indices = requested[
                valid_positions
            ].astype(int)

            raw = self._chl.isel(
                time=valid_indices.tolist(),
                **sl,
            ).load().values.astype(
                np.float32
            )

            time_values = self._ds[
                "time"
            ].isel(
                time=valid_indices.tolist()
            ).values

            for j, position in enumerate(
                valid_positions
            ):
                (
                    values[position],
                    valid[position],
                ) = self._clean_and_normalize(
                    raw[j],
                    ocean_patch,
                )

                timestamp = pd.Timestamp(
                    time_values[j]
                )

                day_of_year = int(
                    timestamp.dayofyear
                )

                angle = (
                    2.0
                    * np.pi
                    * (day_of_year - 1)
                    / 365.25
                )

                sequence_doy[
                    position
                ] = np.array(
                    [
                        np.sin(angle),
                        np.cos(angle),
                    ],
                    dtype=np.float32,
                )

        return (
            values,
            valid,
            time_valid.astype(
                np.float32
            ),
            offsets,
            sequence_doy,
            ocean_patch,
        )

    def _load_mask_patch(
        self,
        time_idx: int,
        y: int,
        x: int,
    ):
        """
        Load only the validity mask for one real observation day.
        """
        self._ensure_open()

        sl = self._patch_slices(y, x)

        raw = self._chl.isel(
            time=int(time_idx),
            **sl,
        ).load().values.astype(np.float32)

        ocean_patch = self.ocean_mask[
            y:y + self.patch_size,
            x:x + self.patch_size,
        ]

        valid = (
            np.isfinite(raw)
            & (raw >= self.chl_min)
            & (raw <= self.chl_max)
            & ocean_patch
        )

        return valid

    # ------------------------------------------------------------------
    # Existing temporal-U-Net context
    # ------------------------------------------------------------------
    def _fill_context(
        self,
        window_values: np.ndarray,
        window_valid: np.ndarray,
        direction: int,
    ):
        """
        Existing temporal-U-Net fallback.

        direction = -1:
            try t-1, then t-2, then t-3

        direction = +1:
            try t+1, then t+2, then t+3
        """
        center = self.max_offset
        h, w = window_values.shape[1:]

        filled = np.zeros(
            (h, w),
            dtype=np.float32,
        )
        source = np.zeros(
            (h, w),
            dtype=np.uint8,
        )
        still_missing = np.ones(
            (h, w),
            dtype=bool,
        )

        for offset in range(
            1,
            self.max_offset + 1,
        ):
            k = center + direction * offset
            use = (
                window_valid[k]
                & still_missing
            )

            filled[use] = (
                window_values[k][use]
            )
            source[use] = offset
            still_missing[use] = False

        return filled, source

    @staticmethod
    def _source_one_hot(
        source: np.ndarray,
        max_offset: int,
    ):
        channels = [
            (source == offset).astype(
                np.float32
            )
            for offset in range(
                1,
                max_offset + 1,
            )
        ]

        return np.stack(
            channels,
            axis=0,
        )

    # ------------------------------------------------------------------
    # Leakage-safe artificial masking
    # ------------------------------------------------------------------
    def _candidate_mask_indices(
        self,
        target_idx: int,
    ):
        candidates = self.mask_time_indices[
            self.mask_time_indices
            != int(target_idx)
        ]

        if len(candidates) == 0:
            raise RuntimeError(
                f"No alternative mask day available "
                f"in split={self.split!r}"
            )

        return candidates

    def _deterministic_eval_mask_order(
        self,
        target_idx: int,
        candidates: np.ndarray,
    ):
        """
        Same target date -> same mask ordering across all patches/runs.
        """
        rng = np.random.default_rng(
            self.seed
            + 1000003 * int(target_idx)
        )
        return rng.permutation(candidates)

    def _choose_artificial_mask(
        self,
        target_idx: int,
        target_valid: np.ndarray,
        y: int,
        x: int,
    ):
        """
        Choose a same-split real observation mask.

        Train:
            random mask selection.

        Validation/test:
            deterministic selection based on target time index.
        """
        candidates = (
            self._candidate_mask_indices(
                target_idx
            )
        )

        if self.split == "train":
            rng = self._get_rng()
            order = rng.permutation(
                candidates
            )
        else:
            order = (
                self._deterministic_eval_mask_order(
                    target_idx,
                    candidates,
                )
            )

        best_mask = None
        best_idx = None
        best_hidden = -1

        attempts = min(
            self.max_mask_attempts,
            len(order),
        )

        for mask_idx in order[:attempts]:
            artificial_mask = (
                self._load_mask_patch(
                    int(mask_idx),
                    y,
                    x,
                )
            )

            hidden_count = int(
                (
                    target_valid
                    & (~artificial_mask)
                ).sum()
            )

            if hidden_count > best_hidden:
                best_hidden = hidden_count
                best_mask = artificial_mask
                best_idx = int(mask_idx)

            if (
                hidden_count
                >= self.min_hidden_pixels
            ):
                break

        if best_mask is None or best_idx is None:
            raise RuntimeError(
                "Could not select an artificial mask."
            )

        return (
            best_mask,
            best_idx,
            best_hidden,
        )

    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        row = self.samples.iloc[int(idx)]

        time_idx = int(
            row["time_idx"]
        )
        y = int(
            row["y"]
        )
        x = int(
            row["x"]
        )
        date = pd.Timestamp(
            row["date"]
        )

        # These are filled only in the modes that use them.
        past = None
        past_source_oh = None
        future = None
        future_source_oh = None

        mamba_values = None
        mamba_valid = None
        mamba_time_valid = None
        mamba_offsets = None
        mamba_doy = None
        aux_current = {}
        aux_mamba = {}

        # --------------------------------------------------------------
        # Load target/context according to model mode
        # --------------------------------------------------------------
        if self.input_mode == "spatial":
            (
                target,
                target_valid,
                ocean_patch,
            ) = self._load_target_patch(
                time_idx,
                y,
                x,
            )

        elif self.input_mode == "temporal":
            (
                window_values,
                window_valid,
                ocean_patch,
            ) = self._load_window_patch(
                time_idx,
                y,
                x,
            )

            center = self.max_offset

            target = (
                window_values[center]
                .copy()
            )
            target_valid = (
                window_valid[center]
                .copy()
            )

            past, past_source = (
                self._fill_context(
                    window_values,
                    window_valid,
                    direction=-1,
                )
            )

            future, future_source = (
                self._fill_context(
                    window_values,
                    window_valid,
                    direction=+1,
                )
            )

            past_source_oh = (
                self._source_one_hot(
                    past_source,
                    self.max_offset,
                )
            )

            future_source_oh = (
                self._source_one_hot(
                    future_source,
                    self.max_offset,
                )
            )

        elif self.input_mode == "direct":
            # DIRECT: load t-3 ... t+3, then build the prioritized
            # previous/future fallback context exactly as described
            # for DIRECT.
            (
                window_values,
                window_valid,
                ocean_patch,
            ) = self._load_window_patch(
                time_idx,
                y,
                x,
            )

            center = self.max_offset

            target = (
                window_values[center]
                .copy()
            )
            target_valid = (
                window_valid[center]
                .copy()
            )

            direct_past, direct_past_source = (
                self._fill_context(
                    window_values,
                    window_valid,
                    direction=-1,
                )
            )

            direct_future, direct_future_source = (
                self._fill_context(
                    window_values,
                    window_valid,
                    direction=+1,
                )
            )

        else:
            # MAMBA: long leakage-safe temporal sequence.
            (
                mamba_values,
                mamba_valid,
                mamba_time_valid,
                mamba_offsets,
                mamba_doy,
                ocean_patch,
            ) = self._load_mamba_sequence_patch(
                time_idx,
                y,
                x,
            )

            center = self.mamba_half_window

            # The central sequence element is always the benchmark target day.
            if not bool(
                mamba_time_valid[center]
            ):
                raise RuntimeError(
                    f"Mamba centre day time_idx={time_idx} "
                    "was marked unavailable."
                )

            target = (
                mamba_values[center]
                .copy()
            )

            target_valid = (
                mamba_valid[center]
                .copy()
            )

        # --------------------------------------------------------------
        # Load selected auxiliary data.
        # --------------------------------------------------------------
        if self.aux_mode != "none":
            aux_current = self._load_aux_current_patch(
                time_idx,
                y,
                x,
            )

            if self.input_mode == "mamba":
                aux_mamba = self._load_aux_mamba_sequence(
                    center_idx=time_idx,
                    y=y,
                    x=x,
                    time_valid=mamba_time_valid,
                    offsets=mamba_offsets,
                )

        # --------------------------------------------------------------
        # Artificially mask known centre-day target values
        # using a real same-split observation mask
        # --------------------------------------------------------------
        (
            artificial_mask,
            mask_time_idx,
            hidden_count,
        ) = self._choose_artificial_mask(
            time_idx,
            target_valid,
            y,
            x,
        )

        current_visible = (
            target_valid
            & artificial_mask
        )

        loss_mask = (
            target_valid
            & (~artificial_mask)
        )

        current_masked = np.zeros_like(
            target,
            dtype=np.float32,
        )

        current_masked[
            current_visible
        ] = target[
            current_visible
        ]

        # BBP443 is an ocean-colour variable and is subject to essentially the
        # same cloud obstruction as CHL. Apply the artificial benchmark mask on
        # the TARGET DAY so BBP443 cannot leak information through simulated
        # clouds. SST is intentionally left untouched.
        if self.use_bbp443 and "bbp443" in aux_current:
            bbp_center_valid = (
                aux_current["bbp443_valid"]
                & artificial_mask
            )
            bbp_center = np.zeros_like(
                aux_current["bbp443"],
                dtype=np.float32,
            )
            bbp_center[bbp_center_valid] = (
                aux_current["bbp443"][bbp_center_valid]
            )
            aux_current["bbp443"] = bbp_center
            aux_current["bbp443_valid"] = bbp_center_valid

            if self.input_mode == "mamba":
                center_aux = self.mamba_half_window
                aux_mamba["bbp443"][center_aux] = bbp_center
                aux_mamba["bbp443_valid"][center_aux] = bbp_center_valid

        # --------------------------------------------------------------
        # Day-of-year seasonal encoding for the centre date
        # --------------------------------------------------------------
        day_of_year = int(
            date.dayofyear
        )

        angle = (
            2.0
            * np.pi
            * (day_of_year - 1)
            / 365.25
        )

        doy = np.array(
            [
                np.sin(angle),
                np.cos(angle),
            ],
            dtype=np.float32,
        )

        # --------------------------------------------------------------
        # Build model input
        # --------------------------------------------------------------
        if self.input_mode == "spatial":
            input_channels = np.stack(
                [
                    current_masked,
                    current_visible.astype(
                        np.float32
                    ),
                    ocean_patch.astype(
                        np.float32
                    ),
                ],
                axis=0,
            ).astype(np.float32)

        elif self.input_mode == "temporal":
            input_channels = np.concatenate(
                [
                    past[None, ...],
                    past_source_oh,
                    current_masked[None, ...],
                    current_visible.astype(
                        np.float32
                    )[None, ...],
                    future[None, ...],
                    future_source_oh,
                    ocean_patch.astype(
                        np.float32
                    )[None, ...],
                ],
                axis=0,
            ).astype(np.float32)

        elif self.input_mode == "direct":
            # ----------------------------------------------------------
            # DIRECT-specific context and masks.
            #
            # direct_context:
            #   [past_filled, current_masked, future_filled]
            #
            # DIRECT source-mask convention from the thesis:
            #   past source 1 -> [0,0,1]
            #   past source 2 -> [0,1,0]
            #   past source 3 -> [1,0,0]
            # and identically for the future source map.
            # ----------------------------------------------------------
            direct_context = np.stack(
                [
                    direct_past,
                    current_masked,
                    direct_future,
                ],
                axis=0,
            ).astype(np.float32)

            direct_past_oh = np.stack(
                [
                    (direct_past_source == 3),
                    (direct_past_source == 2),
                    (direct_past_source == 1),
                ],
                axis=0,
            ).astype(np.float32)

            direct_future_oh = np.stack(
                [
                    (direct_future_source == 3),
                    (direct_future_source == 2),
                    (direct_future_source == 1),
                ],
                axis=0,
            ).astype(np.float32)

            center_observed = current_visible.astype(
                np.float32
            )

            # DIRECT defines the second centre-mask channel as
            # missing OR land, therefore it is simply 1-observed.
            center_missing_or_land = (
                1.0 - center_observed
            ).astype(np.float32)

            direct_temporal_masks = np.concatenate(
                [
                    direct_past_oh,                         # 3
                    center_observed[None, ...],            # 1
                    center_missing_or_land[None, ...],     # 1
                    direct_future_oh,                      # 3
                ],
                axis=0,
            ).astype(np.float32)

            # Convenience tensor for inspection only:
            # 3 context + 8 masks + 1 ocean mask = 12 channels.
            # The actual DIRECT model additionally receives the noisy/FUSE
            # centre field separately, so train_direct.py should normally use
            # direct_context + direct_temporal_masks rather than this tensor.
            input_channels = np.concatenate(
                [
                    direct_context,
                    direct_temporal_masks,
                    ocean_patch.astype(
                        np.float32
                    )[None, ...],
                ],
                axis=0,
            ).astype(np.float32)

        else:
            # ----------------------------------------------------------
            # MAMBA-specific temporal sequence.
            #
            # Natural missing values remain zero and are accompanied by an
            # explicit validity mask. The centre day is overwritten with the
            # exact same artificial benchmark mask used by all other models.
            # ----------------------------------------------------------
            center = self.mamba_half_window

            mamba_chl = (
                mamba_values.copy()
            )

            mamba_obs = (
                mamba_valid.astype(
                    np.float32
                )
            )

            mamba_chl[
                center
            ] = current_masked

            mamba_obs[
                center
            ] = current_visible.astype(
                np.float32
            )

            mamba_ocean = np.broadcast_to(
                ocean_patch.astype(
                    np.float32
                )[None, ...],
                (
                    self.mamba_sequence_length,
                    self.patch_size,
                    self.patch_size,
                ),
            ).copy()

            # A frame outside the split has no usable temporal information.
            # Its ocean channel is zeroed as well so the model can distinguish
            # unavailable sequence positions from ordinary cloudy ocean frames.
            unavailable = (
                mamba_time_valid
                < 0.5
            )

            if unavailable.any():
                mamba_chl[
                    unavailable
                ] = 0.0
                mamba_obs[
                    unavailable
                ] = 0.0
                mamba_ocean[
                    unavailable
                ] = 0.0

            # Channel order is intentionally stable/documented:
            #   CHL value, CHL mask,
            #   [SST value, SST mask],
            #   [BBP443 value, BBP443 mask],
            #   ocean mask
            sequence_channels = [
                mamba_chl,
                mamba_obs,
            ]

            if self.use_sst:
                sst_seq = aux_mamba["sst"].astype(np.float32)
                sst_mask_seq = aux_mamba[
                    "sst_valid"
                ].astype(np.float32)

                if unavailable.any():
                    sst_seq[unavailable] = 0.0
                    sst_mask_seq[unavailable] = 0.0

                sequence_channels.extend(
                    [sst_seq, sst_mask_seq]
                )

            if self.use_bbp443:
                bbp_seq = aux_mamba["bbp443"].astype(np.float32)
                bbp_mask_seq = aux_mamba[
                    "bbp443_valid"
                ].astype(np.float32)

                if unavailable.any():
                    bbp_seq[unavailable] = 0.0
                    bbp_mask_seq[unavailable] = 0.0

                sequence_channels.extend(
                    [bbp_seq, bbp_mask_seq]
                )

            sequence_channels.append(mamba_ocean)

            mamba_sequence = np.stack(
                sequence_channels,
                axis=1,
            ).astype(np.float32)

            # For Mamba mode, input is naturally [T,C,H,W].
            input_channels = mamba_sequence

        # --------------------------------------------------------------
        # Optional auxiliary channels for spatial/temporal/DIRECT inputs.
        #
        # We append only target-day auxiliary fields in these modes. Mamba
        # already receives full auxiliary sequences above.
        # --------------------------------------------------------------
        if self.input_mode != "mamba" and self.aux_mode != "none":
            extra_channels = []

            if self.use_sst:
                extra_channels.extend(
                    [
                        aux_current["sst"][None, ...].astype(np.float32),
                        aux_current["sst_valid"][None, ...].astype(np.float32),
                    ]
                )

            if self.use_bbp443:
                extra_channels.extend(
                    [
                        aux_current["bbp443"][None, ...].astype(np.float32),
                        aux_current["bbp443_valid"][None, ...].astype(np.float32),
                    ]
                )

            if extra_channels:
                input_channels = np.concatenate(
                    [input_channels] + extra_channels,
                    axis=0,
                ).astype(np.float32)

        # --------------------------------------------------------------
        # Common return dictionary
        # --------------------------------------------------------------
        sample = {
            "input": torch.from_numpy(
                input_channels
            ),

            "target": torch.from_numpy(
                target[None, ...].astype(
                    np.float32
                )
            ),

            "current_masked": torch.from_numpy(
                current_masked[None, ...].astype(
                    np.float32
                )
            ),

            "current_visible_mask": torch.from_numpy(
                current_visible.astype(
                    np.float32
                )[None, ...]
            ),

            "loss_mask": torch.from_numpy(
                loss_mask.astype(
                    np.float32
                )[None, ...]
            ),

            "target_valid_mask": torch.from_numpy(
                target_valid.astype(
                    np.float32
                )[None, ...]
            ),

            "ocean_mask": torch.from_numpy(
                ocean_patch.astype(
                    np.float32
                )[None, ...]
            ),

            "doy": torch.from_numpy(
                doy
            ),

            "time_idx": torch.tensor(
                time_idx,
                dtype=torch.long,
            ),

            "mask_time_idx": torch.tensor(
                mask_time_idx,
                dtype=torch.long,
            ),

            "hidden_pixel_count": torch.tensor(
                hidden_count,
                dtype=torch.long,
            ),

            "patch_y": torch.tensor(
                y,
                dtype=torch.long,
            ),

            "patch_x": torch.tensor(
                x,
                dtype=torch.long,
            ),

            "date": date.strftime(
                "%Y-%m-%d"
            ),
        }

        # --------------------------------------------------------------
        # Auxiliary tensors (available in every mode when selected)
        # --------------------------------------------------------------
        if self.use_sst:
            sample.update(
                {
                    "sst": torch.from_numpy(
                        aux_current["sst"][None, ...].astype(np.float32)
                    ),
                    "sst_valid_mask": torch.from_numpy(
                        aux_current["sst_valid"][None, ...].astype(np.float32)
                    ),
                }
            )

        if self.use_bbp443:
            sample.update(
                {
                    "bbp443": torch.from_numpy(
                        aux_current["bbp443"][None, ...].astype(np.float32)
                    ),
                    "bbp443_valid_mask": torch.from_numpy(
                        aux_current["bbp443_valid"][None, ...].astype(np.float32)
                    ),
                }
            )

        # --------------------------------------------------------------
        # Additional DIRECT tensors
        # --------------------------------------------------------------
        if self.input_mode == "direct":
            sample.update(
                {
                    "direct_context": torch.from_numpy(
                        direct_context
                    ),

                    "direct_temporal_masks": torch.from_numpy(
                        direct_temporal_masks
                    ),

                    "past_filled": torch.from_numpy(
                        direct_past[None, ...].astype(
                            np.float32
                        )
                    ),

                    "past_source": torch.from_numpy(
                        direct_past_source.astype(
                            np.int64
                        )
                    ),

                    "future_filled": torch.from_numpy(
                        direct_future[None, ...].astype(
                            np.float32
                        )
                    ),

                    "future_source": torch.from_numpy(
                        direct_future_source.astype(
                            np.int64
                        )
                    ),
                }
            )

        # --------------------------------------------------------------
        # Additional MAMBA tensors
        # --------------------------------------------------------------
        if self.input_mode == "mamba":
            sample.update(
                {
                    "mamba_sequence": torch.from_numpy(
                        mamba_sequence
                    ),

                    "mamba_chl": torch.from_numpy(
                        mamba_chl[
                            :, None, ...
                        ].astype(
                            np.float32
                        )
                    ),

                    "mamba_observation_mask": torch.from_numpy(
                        mamba_obs[
                            :, None, ...
                        ].astype(
                            np.float32
                        )
                    ),

                    "mamba_ocean_mask": torch.from_numpy(
                        mamba_ocean[
                            :, None, ...
                        ].astype(
                            np.float32
                        )
                    ),

                    "mamba_time_valid": torch.from_numpy(
                        mamba_time_valid.astype(
                            np.float32
                        )
                    ),

                    "mamba_offsets": torch.from_numpy(
                        mamba_offsets.astype(
                            np.int64
                        )
                    ),

                    "mamba_doy": torch.from_numpy(
                        mamba_doy.astype(
                            np.float32
                        )
                    ),
                }
            )

            if self.use_sst:
                sample.update(
                    {
                        "mamba_sst": torch.from_numpy(
                            aux_mamba["sst"][:, None, ...].astype(np.float32)
                        ),
                        "mamba_sst_valid_mask": torch.from_numpy(
                            aux_mamba["sst_valid"][:, None, ...].astype(np.float32)
                        ),
                    }
                )

            if self.use_bbp443:
                sample.update(
                    {
                        "mamba_bbp443": torch.from_numpy(
                            aux_mamba["bbp443"][:, None, ...].astype(np.float32)
                        ),
                        "mamba_bbp443_valid_mask": torch.from_numpy(
                            aux_mamba["bbp443_valid"][:, None, ...].astype(np.float32)
                        ),
                    }
                )

        return sample

    def close(self):
        if self._ds is not None:
            self._ds.close()
        if self._sst_ds is not None:
            self._sst_ds.close()
        if self._bbp_ds is not None:
            self._bbp_ds.close()

        self._ds = None
        self._chl = None
        self._spatial_dims = None
        self._sst_ds = None
        self._sst = None
        self._sst_quality = None
        self._bbp_ds = None
        self._bbp = None


def _sanity_check(
    processed_dir: str | Path,
    split: str,
    index: int,
    input_mode: str,
    mamba_sequence_length: int = 15,
    aux_mode: str = "none",
):
    ds = ChlorophyllReconstructionDataset(
        processed_dir=processed_dir,
        split=split,
        input_mode=input_mode,
        mamba_sequence_length=mamba_sequence_length,
        aux_mode=aux_mode,
    )

    print(
        f"Split: {split}"
    )
    print(
        f"Input mode: {input_mode}"
    )
    print(
        f"Samples: {len(ds):,}"
    )
    print(
        f"Aux mode: {ds.aux_mode}"
    )

    sample = ds[index]

    print("\nSample")
    print("------")
    print(
        "date:",
        sample["date"],
    )
    print(
        "input:",
        tuple(
            sample["input"].shape
        ),
    )
    print(
        "target:",
        tuple(
            sample["target"].shape
        ),
    )
    print(
        "current_masked:",
        tuple(
            sample[
                "current_masked"
            ].shape
        ),
    )
    print(
        "current_visible_mask:",
        tuple(
            sample[
                "current_visible_mask"
            ].shape
        ),
    )
    print(
        "loss_mask:",
        tuple(
            sample["loss_mask"].shape
        ),
    )
    print(
        "target_valid_mask:",
        tuple(
            sample[
                "target_valid_mask"
            ].shape
        ),
    )
    print(
        "ocean_mask:",
        tuple(
            sample["ocean_mask"].shape
        ),
    )
    print(
        "doy:",
        sample["doy"].tolist(),
    )
    print(
        "time_idx:",
        int(
            sample["time_idx"]
        ),
    )
    print(
        "mask_time_idx:",
        int(
            sample["mask_time_idx"]
        ),
    )
    print(
        "hidden pixels:",
        int(
            sample[
                "hidden_pixel_count"
            ]
        ),
    )
    print(
        "patch:",
        (
            int(sample["patch_y"]),
            int(sample["patch_x"]),
        ),
    )
    print(
        "input finite:",
        bool(
            torch.isfinite(
                sample["input"]
            ).all()
        ),
    )
    print(
        "target finite:",
        bool(
            torch.isfinite(
                sample["target"]
            ).all()
        ),
    )

    if input_mode != "mamba":
        expected_channels = {
            "spatial": 3,
            "temporal": 11,
            "direct": 12,
        }[input_mode] + 2 * (
            int(ds.use_sst) + int(ds.use_bbp443)
        )

        if (
            sample["input"].shape[0]
            != expected_channels
        ):
            raise AssertionError(
                f"Expected {expected_channels} "
                f"input channels, got "
                f"{sample['input'].shape[0]}"
            )
    else:
        n_aux = int(ds.use_sst) + int(ds.use_bbp443)
        expected_shape = (
            mamba_sequence_length,
            3 + 2 * n_aux,
            ds.patch_size,
            ds.patch_size,
        )

        if tuple(
            sample["input"].shape
        ) != expected_shape:
            raise AssertionError(
                f"Expected Mamba input {expected_shape}, "
                f"got {tuple(sample['input'].shape)}"
            )

    if input_mode == "direct":
        print(
            "direct_context:",
            tuple(
                sample[
                    "direct_context"
                ].shape
            ),
        )
        print(
            "direct_temporal_masks:",
            tuple(
                sample[
                    "direct_temporal_masks"
                ].shape
            ),
        )
        print(
            "past_source:",
            tuple(
                sample["past_source"].shape
            ),
        )
        print(
            "future_source:",
            tuple(
                sample["future_source"].shape
            ),
        )

        if (
            sample[
                "direct_context"
            ].shape[0]
            != 3
        ):
            raise AssertionError(
                "DIRECT context must contain "
                "[past_filled,current_masked,future_filled]."
            )

        if (
            sample[
                "direct_temporal_masks"
            ].shape[0]
            != 8
        ):
            raise AssertionError(
                "DIRECT temporal masks must have 8 channels."
            )

        # Exact centre equality sanity check.
        if not torch.equal(
            sample[
                "direct_context"
            ][1],
            sample[
                "current_masked"
            ][0],
        ):
            raise AssertionError(
                "DIRECT centre frame does "
                "not equal current_masked."
            )

        if not torch.equal(
            sample[
                "direct_temporal_masks"
            ][3],
            sample[
                "current_visible_mask"
            ][0],
        ):
            raise AssertionError(
                "DIRECT observed-centre mask does "
                "not equal current_visible_mask."
            )

    if input_mode == "mamba":
        print(
            "mamba_sequence:",
            tuple(
                sample[
                    "mamba_sequence"
                ].shape
            ),
        )
        print(
            "mamba_time_valid:",
            sample[
                "mamba_time_valid"
            ].tolist(),
        )
        print(
            "mamba_offsets:",
            sample[
                "mamba_offsets"
            ].tolist(),
        )
        print(
            "mamba_doy:",
            tuple(
                sample[
                    "mamba_doy"
                ].shape
            ),
        )

        center = (
            mamba_sequence_length // 2
        )

        if not torch.equal(
            sample[
                "mamba_sequence"
            ][center, 0],
            sample[
                "current_masked"
            ][0],
        ):
            raise AssertionError(
                "Mamba centre CHL does not equal current_masked."
            )

        if not torch.equal(
            sample[
                "mamba_sequence"
            ][center, 1],
            sample[
                "current_visible_mask"
            ][0],
        ):
            raise AssertionError(
                "Mamba centre observation mask does not equal "
                "current_visible_mask."
            )

        if int(
            sample[
                "mamba_offsets"
            ][center]
        ) != 0:
            raise AssertionError(
                "Mamba centre offset must be zero."
            )

    if (
        int(
            sample[
                "hidden_pixel_count"
            ]
        )
        == 0
    ):
        print(
            "\nWARNING: this sample has zero "
            "artificially hidden ground-truth pixels."
        )

    ds.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Sanity-check the chlorophyll "
            "PyTorch Dataset."
        )
    )

    parser.add_argument(
        "--processed-dir",
        required=True,
        help="Path to data/processed",
    )

    parser.add_argument(
        "--split",
        default="train",
        choices=[
            "train",
            "validation",
            "test",
        ],
    )

    parser.add_argument(
        "--input-mode",
        default="temporal",
        choices=[
            "spatial",
            "temporal",
            "direct",
            "mamba",
        ],
    )

    parser.add_argument(
        "--mamba-sequence-length",
        type=int,
        default=15,
        help=(
            "Odd Mamba temporal sequence length. "
            "Default 15 = 7 days before + target + 7 days after."
        ),
    )

    parser.add_argument(
        "--aux-mode",
        default="none",
        choices=[
            "none",
            "sst",
            "bbp443",
            "sst_bbp443",
        ],
        help=(
            "Optional auxiliary inputs. 'none' preserves the exact old "
            "CHL-only channel layout."
        ),
    )

    parser.add_argument(
        "--index",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    _sanity_check(
        processed_dir=args.processed_dir,
        split=args.split,
        index=args.index,
        input_mode=args.input_mode,
        mamba_sequence_length=args.mamba_sequence_length,
        aux_mode=args.aux_mode,
    )

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import glob
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# =========================================================
# Device
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# =========================================================
# Shared settings
# =========================================================
MAX_FRAMES = 300
SPATIAL_START_MODE = "even"

# These were the original-frame plots.
HIGH_RES_REQUESTED_ORIGINAL_FRAMES = [
    40, 50, 60, 70, 80, 90, 100, 110, 120, 150, 180, 296
]

# =========================================================
# Resolution configurations
# =========================================================
TEST_CONFIGS = {
    "high_resolution": {
        "window_frames": 10,
        "temporal_skip_param": 3,
        "target_spatial_size": (301, 301),
        "output_folder": "test_results_fourier_unet3d_local_integral_high_resolution",
        "summary_filename": "high_resolution_relative_squared_l2.txt",
    },
    "low_resolution": {
        "window_frames": 10,
        "temporal_skip_param": 3,
        "target_spatial_size": (121, 121),
        "output_folder": "test_results_fourier_unet3d_local_integral_low_resolution",
        "summary_filename": "low_resolution_relative_squared_l2.txt",
    },
}


# =========================================================
# General helpers
# =========================================================
def skip_data(data: np.ndarray, param: int) -> np.ndarray:
    if param < 0:
        raise ValueError("param must be non-negative")
    return data[::param + 1]


def separate_symmetric_limits(arr):
    vmax = float(np.max(np.abs(arr)))
    if vmax == 0 or np.isnan(vmax):
        vmax = 1e-12
    return -vmax, vmax


# =========================================================
# Spatial downsampling helpers
# =========================================================
def make_downsample_indices(n_original, n_target, start_mode):
    if n_target > n_original:
        raise ValueError(
            f"Target size {n_target} cannot be larger than original size {n_original}"
        )

    if start_mode == "even":
        start = 0
    elif start_mode == "odd":
        start = 1
    else:
        raise ValueError("start_mode must be 'even' or 'odd'")

    indices = np.linspace(start, n_original - 1, n_target)
    indices = np.round(indices).astype(int)
    indices = np.clip(indices, 0, n_original - 1)

    if len(np.unique(indices)) != len(indices):
        raise ValueError(
            "Duplicate spatial indices produced for "
            f"n_original={n_original}, n_target={n_target}, start_mode={start_mode}"
        )

    return indices


def spatial_resample_2d(arr, target_size, start_mode):
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got {arr.shape}")

    target_x, target_y = target_size
    x_indices = make_downsample_indices(arr.shape[0], target_x, start_mode)
    y_indices = make_downsample_indices(arr.shape[1], target_y, start_mode)
    return arr[np.ix_(x_indices, y_indices)]


def spatial_resample_3d_xyt(arr, target_size, start_mode):
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got {arr.shape}")

    target_x, target_y = target_size
    x_indices = make_downsample_indices(arr.shape[0], target_x, start_mode)
    y_indices = make_downsample_indices(arr.shape[1], target_y, start_mode)
    t_indices = np.arange(arr.shape[2])
    return arr[np.ix_(x_indices, y_indices, t_indices)]


def spatial_resample_array(arr, target_size, start_mode):
    if arr.ndim == 2:
        return spatial_resample_2d(arr, target_size, start_mode)
    if arr.ndim == 3:
        return spatial_resample_3d_xyt(arr, target_size, start_mode)
    raise ValueError(f"Expected 2D or 3D array, got {arr.shape}")


# =========================================================
# File discovery
# =========================================================
def get_test_files(data_folder: str):
    data_files = sorted(
        glob.glob(os.path.join(data_folder, "wave_source_*_xyt_kpa.npy"))
    )

    if len(data_files) == 0:
        raise FileNotFoundError(f"No simulation files found in {data_folder}")

    xcoord_file = os.path.join(data_folder, "X_coordinates_mm.npy")
    ycoord_file = os.path.join(data_folder, "Y_coordinates_mm.npy")
    tcoord_file = os.path.join(data_folder, "T_coordinates_us.npy")

    for path in [xcoord_file, ycoord_file, tcoord_file]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required file: {path}")

    return data_files, xcoord_file, ycoord_file, tcoord_file


# =========================================================
# Source-specific files
# =========================================================
def source_index_from_sim(sim_file):
    basename = os.path.basename(sim_file)
    match = re.search(r"wave_source_(\d+)_xyt_kpa\.npy", basename)
    if match is None:
        raise ValueError(f"Could not determine source index from {basename}")
    return match.group(1)


def pulse_files_for_sim(sim_file):
    idx = source_index_from_sim(sim_file)
    folder = os.path.dirname(sim_file)

    pulse_x = os.path.join(folder, f"pulse_source_{idx}_xcoord.npy")
    pulse_y = os.path.join(folder, f"pulse_source_{idx}_ycoord.npy")

    if not os.path.exists(pulse_x):
        raise FileNotFoundError(f"Missing pulse x file: {pulse_x}")
    if not os.path.exists(pulse_y):
        raise FileNotFoundError(f"Missing pulse y file: {pulse_y}")

    return pulse_x, pulse_y


def velocity_file_for_sim(sim_file):
    idx = source_index_from_sim(sim_file)
    folder = os.path.dirname(sim_file)

    source_velocity = os.path.join(folder, f"velocity_map_source_{idx}.npy")
    global_velocity = os.path.join(folder, "velocity_map.npy")

    if os.path.exists(source_velocity):
        return source_velocity

    if os.path.exists(global_velocity):
        print(
            f"Warning: {source_velocity} not found. "
            f"Using fallback {global_velocity}"
        )
        return global_velocity

    raise FileNotFoundError(
        "Missing velocity file. Expected either:\n"
        f"  {source_velocity}\n"
        "or\n"
        f"  {global_velocity}"
    )


# =========================================================
# Time-coordinate preparation
# =========================================================
def prepare_tcoord_xyt(tcoord_raw, simulation_shape):
    """Return time coordinate in shape (x, y, time)."""
    H, W, T = simulation_shape

    if tcoord_raw.ndim == 3:
        if tcoord_raw.shape != simulation_shape:
            raise ValueError(
                f"3D tcoord shape {tcoord_raw.shape} does not match "
                f"simulation shape {simulation_shape}"
            )
        return tcoord_raw

    if tcoord_raw.ndim == 2:
        if tcoord_raw.shape != (H, W):
            raise ValueError(
                f"2D tcoord shape {tcoord_raw.shape} does not match "
                f"spatial shape {(H, W)}"
            )
        return np.tile(tcoord_raw[:, :, None], (1, 1, T))

    raise ValueError(f"Invalid tcoord shape: {tcoord_raw.shape}")


# =========================================================
# Fourier U-Net model definition
# Exact architecture copied from training code
# =========================================================
class SpectralConv3D(nn.Module):
    """
    3D spectral convolution over (time, x, y).

    Input / output shape:
        (B, C, T, H, W)

    Only a limited number of low-frequency Fourier modes are
    learned. The number of modes is clipped automatically at
    each U-Net scale so the block also works after pooling.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        modes_t=8,
        modes_x=16,
        modes_y=16,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_t = modes_t
        self.modes_x = modes_x
        self.modes_y = modes_y

        scale = 1.0 / max(1, in_channels * out_channels)

        # For an rFFT on the last dimension, we keep two
        # temporal and two x-direction corners, while y is
        # represented only by the non-negative rFFT half.
        shape = (
            in_channels,
            out_channels,
            modes_t,
            modes_x,
            modes_y,
        )

        self.weight_pp = nn.Parameter(
            scale * torch.randn(*shape, dtype=torch.cfloat)
        )
        self.weight_np = nn.Parameter(
            scale * torch.randn(*shape, dtype=torch.cfloat)
        )
        self.weight_pn = nn.Parameter(
            scale * torch.randn(*shape, dtype=torch.cfloat)
        )
        self.weight_nn = nn.Parameter(
            scale * torch.randn(*shape, dtype=torch.cfloat)
        )

    @staticmethod
    def compl_mul3d(x, weight):
        # x:      (B, in_channels, mt, mx, my)
        # weight: (in_channels, out_channels, mt, mx, my)
        # output: (B, out_channels, mt, mx, my)
        return torch.einsum(
            "bixyz,ioxyz->boxyz",
            x,
            weight,
        )

    def forward(self, x):
        B, _, T, H, W = x.shape

        x_ft = torch.fft.rfftn(
            x,
            dim=(-3, -2, -1),
            norm="ortho",
        )

        out_ft = torch.zeros(
            B,
            self.out_channels,
            T,
            H,
            W // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        mt = min(self.modes_t, T // 2 if T > 1 else 1)
        mx = min(self.modes_x, H // 2 if H > 1 else 1)
        my = min(self.modes_y, W // 2 + 1)

        # Positive time, positive x.
        out_ft[:, :, :mt, :mx, :my] = self.compl_mul3d(
            x_ft[:, :, :mt, :mx, :my],
            self.weight_pp[:, :, :mt, :mx, :my],
        )

        # Negative time, positive x.
        out_ft[:, :, -mt:, :mx, :my] = self.compl_mul3d(
            x_ft[:, :, -mt:, :mx, :my],
            self.weight_np[:, :, :mt, :mx, :my],
        )

        # Positive time, negative x.
        out_ft[:, :, :mt, -mx:, :my] = self.compl_mul3d(
            x_ft[:, :, :mt, -mx:, :my],
            self.weight_pn[:, :, :mt, :mx, :my],
        )

        # Negative time, negative x.
        out_ft[:, :, -mt:, -mx:, :my] = self.compl_mul3d(
            x_ft[:, :, -mt:, -mx:, :my],
            self.weight_nn[:, :, :mt, :mx, :my],
        )

        return torch.fft.irfftn(
            out_ft,
            s=(T, H, W),
            dim=(-3, -2, -1),
            norm="ortho",
        )


class FourierBlock3D(nn.Module):
    """
    Two consecutive FNO-style Fourier layers.

    Each layer has:
        spectral convolution + 1x1x1 channel mixing
        + GroupNorm + GELU

    The 1x1x1 operation is point-wise channel mixing, not a
    spatial CNN kernel. Therefore there are no 3x3x3 CNN
    convolutions inside the U-Net blocks.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        modes_t=8,
        modes_x=16,
        modes_y=16,
    ):
        super().__init__()

        self.spectral1 = SpectralConv3D(
            in_channels,
            out_channels,
            modes_t=modes_t,
            modes_x=modes_x,
            modes_y=modes_y,
        )
        self.pointwise1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=1,
        )

        self.spectral2 = SpectralConv3D(
            out_channels,
            out_channels,
            modes_t=modes_t,
            modes_x=modes_x,
            modes_y=modes_y,
        )
        self.pointwise2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=1,
        )

        # Same GroupNorm rule as in the CNN U-Net.
        if out_channels >= 8:
            num_groups = 8
        elif out_channels >= 4:
            num_groups = 4
        else:
            num_groups = 1

        self.norm1 = nn.GroupNorm(
            num_groups=num_groups,
            num_channels=out_channels,
        )
        self.norm2 = nn.GroupNorm(
            num_groups=num_groups,
            num_channels=out_channels,
        )

        self.activation = nn.GELU()

    def forward(self, x):
        x = self.activation(
            self.norm1(
                self.spectral1(x) + self.pointwise1(x)
            )
        )
        x = self.activation(
            self.norm2(
                self.spectral2(x) + self.pointwise2(x)
            )
        )
        return x


class LocalIntegralResampler2D(nn.Module):
    """
    Trainable local-integral spatial resampler.

    This is a simplified, separable, DISCO-inspired cross-grid
    local integral operator. It is NOT a verbatim implementation
    of the paper's optimized DISCO layer.

    Input:
        (B, C, T, H_source, W_source)

    Output:
        (B, C, T, H_target, W_target)

    Key properties:
      * Time T is preserved.
      * Channel count C is preserved.
      * The target spatial resolution can differ from the source.
      * The learned kernel is parameterized in normalized physical
        coordinates rather than by a fixed number of pixels.
      * The same learned parameters can therefore be evaluated on
        different source resolutions.

    The continuous local kernel is represented by triangular basis
    functions over normalized distance d in [0, radius]:

        k_theta(d) = sum_l theta_l phi_l(d)

    A separable 2-D integral is then approximated along H and W.
    """

    def __init__(
        self,
        channels,
        target_size=None,
        radius=0.08,
        num_basis=8,
    ):
        super().__init__()

        self.channels = channels

        if target_size is None:
            self.target_size = None
        elif isinstance(target_size, int):
            self.target_size = (target_size, target_size)
        else:
            self.target_size = tuple(target_size)

        self.radius = float(radius)
        self.num_basis = int(num_basis)

        if self.radius <= 0.0:
            raise ValueError("radius must be > 0")

        if self.num_basis < 2:
            raise ValueError("num_basis must be >= 2")

        # One learned continuous 1-D kernel per feature channel.
        #
        # Initialize with a smooth, positive, center-weighted shape.
        # Training is free to move the coefficients positive/negative.
        centers = torch.linspace(
            0.0,
            1.0,
            self.num_basis,
        )

        init = torch.exp(
            -0.5 * (centers / 0.45) ** 2
        )

        self.theta = nn.Parameter(
            init[None, :].repeat(
                channels,
                1,
            )
        )

    def _basis(self, distance):
        """
        distance:
            (...,) normalized physical distance

        return:
            (..., num_basis)
        """

        centers = torch.linspace(
            0.0,
            self.radius,
            self.num_basis,
            device=distance.device,
            dtype=distance.dtype,
        )

        width = (
            self.radius
            / (self.num_basis - 1)
        )

        phi = (
            1.0
            - torch.abs(
                distance[..., None]
                - centers
            )
            / (width + 1e-12)
        )

        return torch.clamp(
            phi,
            min=0.0,
        )

    def _make_axis_weights(
        self,
        n_source,
        n_target,
        device,
        dtype,
    ):
        """
        Build channel-specific source -> target weights.

        Returns:
            weights with shape:
                (C, n_target, n_source)
        """

        if n_source < 1 or n_target < 1:
            raise ValueError(
                "Source and target sizes must be positive."
            )

        # Normalized physical coordinate across the same domain.
        source_coord = torch.linspace(
            0.0,
            1.0,
            n_source,
            device=device,
            dtype=dtype,
        )

        target_coord = torch.linspace(
            0.0,
            1.0,
            n_target,
            device=device,
            dtype=dtype,
        )

        # Pairwise physical distance:
        #     (target, source)
        distance = torch.abs(
            target_coord[:, None]
            - source_coord[None, :]
        )

        phi = self._basis(distance)
        # (target, source, basis)

        # Channel-specific continuous-kernel values:
        #
        # theta: (C, basis)
        # phi:   (target, source, basis)
        #
        # -> (C, target, source)
        weights = torch.einsum(
            "cl,tsl->cts",
            self.theta.to(dtype=dtype),
            phi,
        )

        # Strictly local physical support.
        support = (
            distance <= self.radius
        ).to(dtype)

        weights = (
            weights
            * support[None, :, :]
        )

        # Numerical quadrature factor for a uniform source grid.
        if n_source > 1:
            q = 1.0 / (n_source - 1)
        else:
            q = 1.0

        weights = weights * q

        # Keep the quadrature-scaled kernel weights directly.
        #
        # IMPORTANT:
        # We intentionally do NOT apply L1 normalization here.
        # The factor q = 1 / (n_source - 1) approximates the
        # grid spacing in the normalized [0,1] domain, so keeping
        # it preserves the quadrature-style local integral:
        #
        #     integral k(x,y) u(y) dy
        #       ~= sum_j k(x,y_j) u(y_j) q
        #
        # Normalizing the weights afterward would cancel q and
        # turn the layer into a normalized weighted aggregation.

        return weights

    def forward(
        self,
        x,
        target_size=None,
    ):
        """
        target_size:
            optional dynamic (H_target, W_target).

        This is used for the final decoder resampler so that
        60 -> 121 during training, but can later become
        60 -> 301 during high-resolution inference.
        """

        B, C, T, H, W = x.shape

        if C != self.channels:
            raise RuntimeError(
                f"Expected {self.channels} channels, got {C}"
            )

        if target_size is None:
            if self.target_size is None:
                raise ValueError(
                    "target_size must be provided for a "
                    "dynamic LocalIntegralResampler2D."
                )

            H_target, W_target = self.target_size
        else:
            H_target, W_target = target_size

        # H direction:
        #   (C, H_target, H_source)
        Wh = self._make_axis_weights(
            H,
            H_target,
            x.device,
            x.dtype,
        )

        # W direction:
        #   (C, W_target, W_source)
        Ww = self._make_axis_weights(
            W,
            W_target,
            x.device,
            x.dtype,
        )

        # Apply separable local integral in H:
        #
        # x  : B C T H W
        # Wh : C A H
        #
        # -> B C T A W
        x = torch.einsum(
            "cah,bcthw->bctaw",
            Wh,
            x,
        )

        # Apply separable local integral in W:
        #
        # Ww : C D W
        #
        # -> B C T A D
        x = torch.einsum(
            "cdw,bctaw->bctad",
            Ww,
            x,
        )

        return x


class FourierUNet3D(nn.Module):
    """
    Fourier U-Net with learned local-integral spatial resampling.

    Fourier/FNO-style blocks are unchanged from the previous model.

    The only architectural replacement is:

        MaxPool3D
            ->
        LocalIntegralResampler2D

    and

        trilinear decoder resize
            ->
        LocalIntegralResampler2D

    Internal spatial grids are fixed:

        native HxW
            -> 60x60
            -> 30x30
            -> 15x15
            -> 30x30
            -> 60x60
            -> native HxW

    Therefore a later 301x301 test can follow:

        301 -> 60 -> 30 -> 15 -> 30 -> 60 -> 301

    while the training grid follows:

        121 -> 60 -> 30 -> 15 -> 30 -> 60 -> 121

    Time is never resampled.
    """

    def __init__(
        self,
        in_channels=7,
        out_channels=1,
        base_channels=16,
        modes_t=8,
        modes_x=16,
        modes_y=16,
        grid1=60,
        grid2=30,
        grid3=15,
        integral_radius=0.08,
        integral_num_basis=8,
    ):
        super().__init__()

        c1 = base_channels   # 16
        c2 = 24
        c3 = 32
        c4 = 48

        self.grid1 = int(grid1)
        self.grid2 = int(grid2)
        self.grid3 = int(grid3)

        # -------------------------
        # Encoder Fourier blocks
        # -------------------------
        self.encoder1 = FourierBlock3D(
            in_channels,
            c1,
            modes_t,
            modes_x,
            modes_y,
        )

        self.encoder2 = FourierBlock3D(
            c1,
            c2,
            modes_t,
            modes_x,
            modes_y,
        )

        self.encoder3 = FourierBlock3D(
            c2,
            c3,
            modes_t,
            modes_x,
            modes_y,
        )

        self.bottleneck = FourierBlock3D(
            c3,
            c4,
            modes_t,
            modes_x,
            modes_y,
        )

        # -----------------------------------------------------
        # Learned local-integral downsampling.
        #
        # Channels do not change here.
        # Time does not change here.
        # -----------------------------------------------------
        self.down1 = LocalIntegralResampler2D(
            channels=c1,
            target_size=(
                self.grid1,
                self.grid1,
            ),
            radius=integral_radius,
            num_basis=integral_num_basis,
        )

        self.down2 = LocalIntegralResampler2D(
            channels=c2,
            target_size=(
                self.grid2,
                self.grid2,
            ),
            radius=integral_radius,
            num_basis=integral_num_basis,
        )

        self.down3 = LocalIntegralResampler2D(
            channels=c3,
            target_size=(
                self.grid3,
                self.grid3,
            ),
            radius=integral_radius,
            num_basis=integral_num_basis,
        )

        # -------------------------
        # Decoder Fourier blocks
        # -------------------------
        self.decoder3 = FourierBlock3D(
            c4 + c3,
            c3,
            modes_t,
            modes_x,
            modes_y,
        )

        self.decoder2 = FourierBlock3D(
            c3 + c2,
            c2,
            modes_t,
            modes_x,
            modes_y,
        )

        self.decoder1 = FourierBlock3D(
            c2 + c1,
            c1,
            modes_t,
            modes_x,
            modes_y,
        )

        # -----------------------------------------------------
        # Learned local-integral upsampling.
        #
        # up3: 15 -> 30
        # up2: 30 -> 60
        # up1: 60 -> native HxW dynamically
        # -----------------------------------------------------
        self.up3 = LocalIntegralResampler2D(
            channels=c4,
            target_size=(
                self.grid2,
                self.grid2,
            ),
            radius=integral_radius,
            num_basis=integral_num_basis,
        )

        self.up2 = LocalIntegralResampler2D(
            channels=c3,
            target_size=(
                self.grid1,
                self.grid1,
            ),
            radius=integral_radius,
            num_basis=integral_num_basis,
        )

        self.up1 = LocalIntegralResampler2D(
            channels=c2,
            target_size=None,
            radius=integral_radius,
            num_basis=integral_num_basis,
        )

        # Point-wise output projection.
        self.output_layer = nn.Conv3d(
            c1,
            out_channels,
            kernel_size=1,
        )

    def forward(self, x):
        # Save the incoming native spatial resolution.
        _, _, _, H_native, W_native = x.shape

        # -------------------------
        # Encoder level 1
        # native HxW
        # -------------------------
        enc1 = self.encoder1(x)

        # -------------------------
        # Integral downsample:
        # native HxW -> 60x60
        # -------------------------
        x = self.down1(enc1)
        enc2 = self.encoder2(x)

        # -------------------------
        # Integral downsample:
        # 60x60 -> 30x30
        # -------------------------
        x = self.down2(enc2)
        enc3 = self.encoder3(x)

        # -------------------------
        # Integral downsample:
        # 30x30 -> 15x15
        # -------------------------
        x = self.down3(enc3)
        bottleneck = self.bottleneck(x)

        # -------------------------
        # Decoder level 3
        # 15x15 -> 30x30
        # -------------------------
        dec3 = self.up3(bottleneck)

        if dec3.shape[2:] != enc3.shape[2:]:
            raise RuntimeError(
                "up3 shape does not match skip 3: "
                f"{tuple(dec3.shape)} vs {tuple(enc3.shape)}"
            )

        dec3 = torch.cat(
            [dec3, enc3],
            dim=1,
        )

        dec3 = self.decoder3(dec3)

        # -------------------------
        # Decoder level 2
        # 30x30 -> 60x60
        # -------------------------
        dec2 = self.up2(dec3)

        if dec2.shape[2:] != enc2.shape[2:]:
            raise RuntimeError(
                "up2 shape does not match skip 2: "
                f"{tuple(dec2.shape)} vs {tuple(enc2.shape)}"
            )

        dec2 = torch.cat(
            [dec2, enc2],
            dim=1,
        )

        dec2 = self.decoder2(dec2)

        # -------------------------
        # Decoder level 1
        # 60x60 -> native HxW
        # -------------------------
        dec1 = self.up1(
            dec2,
            target_size=(
                H_native,
                W_native,
            ),
        )

        if dec1.shape[2:] != enc1.shape[2:]:
            raise RuntimeError(
                "up1 shape does not match skip 1: "
                f"{tuple(dec1.shape)} vs {tuple(enc1.shape)}"
            )

        dec1 = torch.cat(
            [dec1, enc1],
            dim=1,
        )

        dec1 = self.decoder1(dec1)

        return self.output_layer(dec1)

# =========================================================
# Model loading
# =========================================================
def load_model(model_file: str, config_file: str = None):
    """
    Reconstruct the Fourier U-Net + local-integral resampling model
    exactly as in training.

    Defaults match the supplied training code. If the saved config
    exists, its values override these defaults.
    """
    config = {
        "in_channels": 7,
        "out_channels": 1,
        "base_channels": 16,
        "modes_t": 24,
        "modes_x": 24,
        "modes_y": 24,
        "window_size": 10,
        "skip_param": 3,
        "loss_crop_size": 121,
        "integral_grid_1": 60,
        "integral_grid_2": 30,
        "integral_grid_3": 15,
        "integral_radius": 0.08,
        "integral_num_basis": 8,
    }

    if config_file is not None and os.path.exists(config_file):
        saved_config = np.load(config_file, allow_pickle=True).item()
        for key in config:
            if key in saved_config:
                config[key] = saved_config[key]
        print(f"Loaded model configuration from: {config_file}")
        print("Saved configuration:", saved_config)
    else:
        print(
            "Model config file not found; using architecture "
            "settings from the supplied training code."
        )

    model = FourierUNet3D(
        in_channels=int(config["in_channels"]),
        out_channels=int(config["out_channels"]),
        base_channels=int(config["base_channels"]),
        modes_t=int(config["modes_t"]),
        modes_x=int(config["modes_x"]),
        modes_y=int(config["modes_y"]),
        grid1=int(config["integral_grid_1"]),
        grid2=int(config["integral_grid_2"]),
        grid3=int(config["integral_grid_3"]),
        integral_radius=float(config["integral_radius"]),
        integral_num_basis=int(config["integral_num_basis"]),
    ).to(device)

    state = torch.load(
        model_file,
        map_location=device,
        weights_only=False,
    )

    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    model.load_state_dict(state)
    model.eval()

    num_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(f"Loaded Fourier U-Net + Local Integral model from: {model_file}")
    print(f"Trainable parameters: {num_parameters:,}")
    print(
        "Local-integral hierarchy: native -> "
        f"{config['integral_grid_1']} -> {config['integral_grid_2']} -> "
        f"{config['integral_grid_3']} -> {config['integral_grid_2']} -> "
        f"{config['integral_grid_1']} -> native"
    )

    return model, config


# =========================================================
# Autoregressive evaluation for one resolution
# =========================================================
def evaluate_autoregressive(
    sim_file,
    xcoord_file,
    ycoord_file,
    tcoord_file,
    model,
    root_out_dir,
    resolution_name,
    window_frames,
    temporal_skip_param,
    target_spatial_size,
):
    sim_name = os.path.splitext(os.path.basename(sim_file))[0]
    save_dir = os.path.join(root_out_dir, sim_name)
    os.makedirs(save_dir, exist_ok=True)

    span = window_frames

    print("\n===================================")
    print(f"Evaluating {sim_name} | {resolution_name}")
    print("===================================")

    # -----------------------------------------------------
    # Load full-resolution wave (x, y, time)
    # -----------------------------------------------------
    sim_full_xyt = np.load(sim_file).astype(np.float32)
    if sim_full_xyt.ndim != 3:
        raise ValueError(f"Expected wave shape (x,y,t), got {sim_full_xyt.shape}")

    original_H, original_W, original_T = sim_full_xyt.shape
    original_T_used = min(MAX_FRAMES, original_T)
    sim_full_xyt = sim_full_xyt[:, :, :original_T_used]

    print("Original wave shape used:", sim_full_xyt.shape)

    # -----------------------------------------------------
    # Load conditioning maps
    # -----------------------------------------------------
    velocity_file = velocity_file_for_sim(sim_file)
    pulse_x_file, pulse_y_file = pulse_files_for_sim(sim_file)

    velocity_full = np.load(velocity_file).astype(np.float32)
    pulse_x_full = np.load(pulse_x_file).astype(np.float32)
    pulse_y_full = np.load(pulse_y_file).astype(np.float32)
    xcoord_full = np.load(xcoord_file).astype(np.float32)
    ycoord_full = np.load(ycoord_file).astype(np.float32)
    tcoord_raw = np.load(tcoord_file).astype(np.float32)

    for name, arr in [
        ("velocity", velocity_full),
        ("pulse_x", pulse_x_full),
        ("pulse_y", pulse_y_full),
        ("xcoord", xcoord_full),
        ("ycoord", ycoord_full),
    ]:
        if arr.shape != (original_H, original_W):
            raise ValueError(
                f"{name} shape {arr.shape} does not match wave spatial shape "
                f"{(original_H, original_W)}"
            )

    tcoord_full_xyt = prepare_tcoord_xyt(
        tcoord_raw,
        (original_H, original_W, original_T),
    )[:, :, :original_T_used]

    # -----------------------------------------------------
    # Spatial resolution handling
    # 301 -> 301 for high resolution (all points retained)
    # 301 -> 121 for low resolution
    # -----------------------------------------------------
    sim_spatial_xyt = spatial_resample_array(
        sim_full_xyt, target_spatial_size, SPATIAL_START_MODE
    ).astype(np.float32)
    velocity_spatial = spatial_resample_array(
        velocity_full, target_spatial_size, SPATIAL_START_MODE
    ).astype(np.float32)
    pulse_x_spatial = spatial_resample_array(
        pulse_x_full, target_spatial_size, SPATIAL_START_MODE
    ).astype(np.float32)
    pulse_y_spatial = spatial_resample_array(
        pulse_y_full, target_spatial_size, SPATIAL_START_MODE
    ).astype(np.float32)
    xcoord_spatial = spatial_resample_array(
        xcoord_full, target_spatial_size, SPATIAL_START_MODE
    ).astype(np.float32)
    ycoord_spatial = spatial_resample_array(
        ycoord_full, target_spatial_size, SPATIAL_START_MODE
    ).astype(np.float32)
    tcoord_spatial_xyt = spatial_resample_array(
        tcoord_full_xyt, target_spatial_size, SPATIAL_START_MODE
    ).astype(np.float32)

    # -----------------------------------------------------
    # Convert (x,y,t) -> (t,x,y), then temporal skipping
    # -----------------------------------------------------
    sim = np.transpose(sim_spatial_xyt, (2, 0, 1))
    tcoord = np.transpose(tcoord_spatial_xyt, (2, 0, 1))

    sim = skip_data(sim, temporal_skip_param).astype(np.float32)
    tcoord = skip_data(tcoord, temporal_skip_param).astype(np.float32)

    T, H, W = sim.shape

    if (H, W) != target_spatial_size:
        raise RuntimeError(
            f"Expected spatial size {target_spatial_size}, got {(H, W)}"
        )

    if tcoord.shape != sim.shape:
        raise ValueError(
            f"tcoord shape {tcoord.shape} does not match wave shape {sim.shape}"
        )

    represented_original_indices = np.arange(
        0,
        original_T_used,
        temporal_skip_param + 1,
    )[:T]

    print("Final model-resolution wave shape:", sim.shape)
    print("Temporal skip parameter:", temporal_skip_param)
    print("Window size:", window_frames)
    print("First represented original indices:", represented_original_indices[:20])

    if T < 2 * window_frames:
        raise ValueError(
            f"Only {T} effective frames remain, but at least "
            f"{2 * window_frames} are required."
        )

    # -----------------------------------------------------
    # Repeat static maps across effective time
    # -----------------------------------------------------
    velocity = np.tile(velocity_spatial[None, :, :], (T, 1, 1))
    pulse_x = np.tile(pulse_x_spatial[None, :, :], (T, 1, 1))
    pulse_y = np.tile(pulse_y_spatial[None, :, :], (T, 1, 1))
    xcoord = np.tile(xcoord_spatial[None, :, :], (T, 1, 1))
    ycoord = np.tile(ycoord_spatial[None, :, :], (T, 1, 1))

    def to_5d_tensor(arr):
        return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)

    sim_t = to_5d_tensor(sim)
    velocity_t = to_5d_tensor(velocity)
    pulse_x_t = to_5d_tensor(pulse_x)
    pulse_y_t = to_5d_tensor(pulse_y)
    xcoord_t = to_5d_tensor(xcoord)
    ycoord_t = to_5d_tensor(ycoord)
    tcoord_t = to_5d_tensor(tcoord)

    # -----------------------------------------------------
    # Initial input
    # IMPORTANT: channel order is kept exactly as training:
    # [wave, velocity, pulse_x, pulse_y, xcoord, ycoord, tcoord]
    # -----------------------------------------------------
    current_wave = sim_t[:, :, :window_frames]

    current_input = torch.cat(
        [
            current_wave,
            velocity_t[:, :, :window_frames],
            pulse_x_t[:, :, :window_frames],
            pulse_y_t[:, :, :window_frames],
            xcoord_t[:, :, :window_frames],
            ycoord_t[:, :, :window_frames],
            tcoord_t[:, :, :window_frames],
        ],
        dim=1,
    )

    expected_input_shape = (1, 7, window_frames, H, W)
    if tuple(current_input.shape) != expected_input_shape:
        raise RuntimeError(
            f"Expected input shape {expected_input_shape}, "
            f"got {tuple(current_input.shape)}"
        )

    print("Initial input tensor shape:", tuple(current_input.shape))

    predictions = [sim[:window_frames]]
    current_start = 0
    step = 0

    # -----------------------------------------------------
    # Autoregressive rollout
    # -----------------------------------------------------
    with torch.inference_mode():
        while current_start + window_frames < T:
            print(
                f"AR step {step}: effective start={current_start}, "
                f"input={tuple(current_input.shape)}",
                flush=True,
            )

            output = model(current_input)

            expected_output_shape = (1, 1, window_frames, H, W)
            if tuple(output.shape) != expected_output_shape:
                raise RuntimeError(
                    f"Expected output shape {expected_output_shape}, "
                    f"got {tuple(output.shape)}"
                )

            span_effective = min(
                span,
                T - (current_start + window_frames),
            )

            next_prediction = output[:, :, :span_effective]

            predictions.append(
                next_prediction[0, 0].cpu().numpy().astype(np.float32)
            )

            old_wave = current_input[:, 0:1, span_effective:]
            new_wave = torch.cat([old_wave, next_prediction], dim=2)

            new_start = current_start + span_effective
            new_end = new_start + window_frames

            def get_window(tensor):
                window = tensor[:, :, new_start:new_end]

                if window.shape[2] < window_frames:
                    missing = window_frames - window.shape[2]

                    if window.shape[2] == 0:
                        raise RuntimeError("Cannot pad an empty conditioning window.")

                    padding = window[:, :, -1:].repeat(
                        1, 1, missing, 1, 1
                    )
                    window = torch.cat([window, padding], dim=2)

                return window

            # Channel order remains identical to training.
            current_input = torch.cat(
                [
                    new_wave,
                    get_window(velocity_t),
                    get_window(pulse_x_t),
                    get_window(pulse_y_t),
                    get_window(xcoord_t),
                    get_window(ycoord_t),
                    get_window(tcoord_t),
                ],
                dim=1,
            )

            current_start += span_effective
            step += 1

    rollout = np.concatenate(predictions, axis=0)[:T]

    if rollout.shape != sim.shape:
        raise RuntimeError(
            f"Rollout shape {rollout.shape} does not match ground truth {sim.shape}"
        )

    # -----------------------------------------------------
    # Relative squared L2 error
    #
    # Main metric:
    #     sum((prediction - ground_truth)^2)
    #     -----------------------------------
    #          sum(ground_truth^2)
    #
    # The sums are taken over ALL predicted frames and ALL
    # spatial points together. The initial input window is
    # excluded because those frames are ground-truth inputs,
    # not model predictions.
    # -----------------------------------------------------
    prediction_eval = rollout[window_frames:]
    ground_truth_eval = sim[window_frames:]

    squared_error = np.sum(
        (prediction_eval - ground_truth_eval) ** 2,
        dtype=np.float64,
    )
    ground_truth_energy = np.sum(
        ground_truth_eval ** 2,
        dtype=np.float64,
    )

    if ground_truth_energy <= 0.0:
        relative_squared_l2 = np.nan
    else:
        relative_squared_l2 = float(
            squared_error / ground_truth_energy
        )

    # Frame-wise relative squared L2 error is kept only for
    # the error-vs-time plot.
    frame_squared_errors = np.sum(
        (prediction_eval - ground_truth_eval) ** 2,
        axis=(1, 2),
        dtype=np.float64,
    )
    frame_ground_truth_energy = np.sum(
        ground_truth_eval ** 2,
        axis=(1, 2),
        dtype=np.float64,
    )

    frame_relative_squared_l2 = np.divide(
        frame_squared_errors,
        frame_ground_truth_energy,
        out=np.full_like(
            frame_squared_errors,
            np.nan,
            dtype=np.float64,
        ),
        where=frame_ground_truth_energy > 0.0,
    )

    predicted_original_indices = represented_original_indices[window_frames:]

    print(
        f"Relative squared L2 error ({resolution_name}): "
        f"{relative_squared_l2:.6e}"
    )

    # -----------------------------------------------------
    # Per-simulation summary
    # -----------------------------------------------------
    with open(
        os.path.join(save_dir, "relative_squared_l2.txt"),
        "w",
        encoding="utf-8",
    ) as file:
        file.write(f"Simulation: {sim_name}\n")
        file.write(f"Resolution test: {resolution_name}\n")
        file.write(f"Velocity file: {os.path.basename(velocity_file)}\n")
        file.write(f"Spatial resolution: {H} x {W}\n")
        file.write(f"Spatial start mode: {SPATIAL_START_MODE}\n")
        file.write(f"Temporal skip parameter: {temporal_skip_param}\n")
        file.write(f"Window size: {window_frames}\n")
        file.write(f"Effective frames: {T}\n")
        file.write(f"Relative squared L2 error: {relative_squared_l2:.10e}\n")

    # -----------------------------------------------------
    # Save arrays
    # -----------------------------------------------------
    np.save(
        os.path.join(save_dir, f"{sim_name}_prediction_{H}x{W}.npy"),
        rollout,
    )
    np.save(
        os.path.join(save_dir, f"{sim_name}_ground_truth_{H}x{W}.npy"),
        sim,
    )
    np.save(
        os.path.join(save_dir, "represented_original_frame_indices.npy"),
        represented_original_indices,
    )
    np.save(
        os.path.join(save_dir, "frame_relative_squared_l2.npy"),
        frame_relative_squared_l2,
    )

    # -----------------------------------------------------
    # Frame-wise relative squared L2 curve versus ORIGINAL
    # frame index
    # -----------------------------------------------------
    plt.figure(figsize=(10, 4))
    plt.plot(predicted_original_indices, frame_relative_squared_l2)
    plt.xlabel("Original simulation frame index")
    plt.ylabel("Frame-wise relative squared L2 error")
    plt.title(
        f"{sim_name} | Fourier U-Net + Local Integral AR "
        f"relative squared L2 | {resolution_name}"
    )
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            save_dir,
            "relative_squared_l2_vs_original_time_index.png",
        ),
        dpi=150,
    )
    plt.close()

    # -----------------------------------------------------
    # GT vs prediction plots at exactly the SAME original
    # simulation frames for both resolutions.
    # -----------------------------------------------------
    original_to_effective = {
        int(original_index): effective_index
        for effective_index, original_index in enumerate(represented_original_indices)
    }

    for original_frame in COMMON_ORIGINAL_FRAMES_TO_PLOT:
        if original_frame not in original_to_effective:
            raise RuntimeError(
                f"Common original frame {original_frame} is not represented "
                f"in {resolution_name}."
            )

        effective_frame = original_to_effective[original_frame]
        ground_truth = sim[effective_frame]
        prediction = rollout[effective_frame]

        frame_squared_error = np.sum(
            (prediction - ground_truth) ** 2,
            dtype=np.float64,
        )
        frame_ground_truth_energy_single = np.sum(
            ground_truth ** 2,
            dtype=np.float64,
        )

        if frame_ground_truth_energy_single <= 0.0:
            frame_relative_squared_l2_single = np.nan
        else:
            frame_relative_squared_l2_single = float(
                frame_squared_error / frame_ground_truth_energy_single
            )

        # Use ground-truth limits for BOTH panels.
        vmin, vmax = separate_symmetric_limits(ground_truth)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        image_gt = axes[0].imshow(
            ground_truth.T,
            origin="lower",
            cmap="seismic",
            vmin=vmin,
            vmax=vmax,
        )
        axes[0].set_title(
            f"Ground truth\noriginal frame {original_frame} "
            f"(effective index {effective_frame})"
        )
        axes[0].axis("off")
        plt.colorbar(image_gt, ax=axes[0])

        image_prediction = axes[1].imshow(
            prediction.T,
            origin="lower",
            cmap="seismic",
            vmin=vmin,
            vmax=vmax,
        )
        axes[1].set_title(
            f"Fourier U-Net + Local Integral prediction\noriginal frame {original_frame} "
            f"(effective index {effective_frame})"
        )
        axes[1].axis("off")
        plt.colorbar(image_prediction, ax=axes[1])

        plt.suptitle(
            f"{sim_name} | {resolution_name} | frame relative squared L2="
            f"{frame_relative_squared_l2_single:.3e}"
        )
        plt.tight_layout()

        plt.savefig(
            os.path.join(
                save_dir,
                f"{sim_name}_original_frame_{original_frame:04d}_gt_vs_prediction.png",
            ),
            dpi=150,
        )
        plt.close()

    return relative_squared_l2


# =========================================================
# Save one overall relative squared L2 summary per resolution
# =========================================================
def save_resolution_summary(results, output_dir, summary_filename, config):
    summary_path = os.path.join(output_dir, summary_filename)

    with open(summary_path, "w", encoding="utf-8") as file:
        file.write(f"Resolution: {config['target_spatial_size'][0]} x {config['target_spatial_size'][1]}\n")
        file.write(f"Window frames: {config['window_frames']}\n")
        file.write(f"Temporal skip parameter: {config['temporal_skip_param']}\n")
        file.write("\nPer-simulation relative squared L2 error:\n")

        for name, rel_error in results.items():
            file.write(f"{name}: {rel_error:.10e}\n")

        if len(results) > 0:
            overall_mean = float(np.mean(list(results.values())))
            file.write(f"\nMean relative squared L2 error across simulations: {overall_mean:.10e}\n")
        else:
            overall_mean = np.nan
            file.write("\nMean relative squared L2 error across simulations: NaN\n")

    print(f"Saved overall relative squared L2 summary: {summary_path}")
    return overall_mean


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    data_folder = os.path.join(script_dir, "all_test_data")
    model_path = os.path.join(script_dir, "fourier_unet3d_local_integral_model.pth")
    config_path = os.path.join(script_dir, "fourier_unet3d_local_integral_config.npy")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Fourier U-Net + Local Integral model not found: {model_path}")

    (
        sim_files,
        xcoord_file,
        ycoord_file,
        tcoord_file,
    ) = get_test_files(data_folder)

    print("\nSimulation files:")
    for sim_file in sim_files:
        print(" ", os.path.basename(sim_file))

    # Load the SAME trained Fourier U-Net + Local Integral model once and use it for both resolutions.
    model, model_config = load_model(model_path, config_path)

    print(
        "\nTraining window size recorded in config:",
        model_config.get("window_size", 10),
    )
    print(
        "Testing uses 10 frames for low resolution and "
        "40 frames for high resolution to represent the same "
        "physical time span as in the reference test."
    )
    print(
        "The Fourier U-Net + Local Integral model can accept both temporal lengths "
        "because its pooling is spatial-only and its spectral "
        "mode count is clipped to the available tensor size."
    )

    all_resolution_means = {}

    for resolution_name, config in TEST_CONFIGS.items():
        print("\n\n########################################################")
        print(f"STARTING {resolution_name.upper()} TEST")
        print("########################################################")

        root_out_dir = os.path.join(script_dir, config["output_folder"])
        os.makedirs(root_out_dir, exist_ok=True)

        results = {}

        for sim_file in sim_files:
            sim_key = os.path.basename(sim_file)

            rel_error = evaluate_autoregressive(
                sim_file=sim_file,
                xcoord_file=xcoord_file,
                ycoord_file=ycoord_file,
                tcoord_file=tcoord_file,
                model=model,
                root_out_dir=root_out_dir,
                resolution_name=resolution_name,
                window_frames=config["window_frames"],
                temporal_skip_param=config["temporal_skip_param"],
                target_spatial_size=config["target_spatial_size"],
            )

            results[sim_key] = rel_error

        overall_mean = save_resolution_summary(
            results=results,
            output_dir=root_out_dir,
            summary_filename=config["summary_filename"],
            config=config,
        )

        all_resolution_means[resolution_name] = overall_mean

        print(f"\nCompleted {resolution_name} test.")
        print("Per-simulation relative squared L2 errors:")
        for name, rel_error in results.items():
            print(f"  {name}: {rel_error:.6e}")
        print(f"Mean relative squared L2 error across simulations: {overall_mean:.6e}")

    print("\n\n========================================================")
    print("ALL FOURIER U-NET + LOCAL INTEGRAL RESOLUTION TESTS COMPLETED")
    print("========================================================")
    for resolution_name, mean_rel_error in all_resolution_means.items():
        print(
            f"{resolution_name}: mean relative squared L2 error = "
            f"{mean_rel_error:.6e}"
        )

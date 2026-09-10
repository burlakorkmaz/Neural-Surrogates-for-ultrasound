import os
import glob
import time
import math
import datetime

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
import matplotlib.pyplot as plt


# =========================================================
# Device
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")


# =========================================================
# Results directory
# =========================================================
now = datetime.datetime.now()
timestamp = now.strftime("%d%m%Y_%H%M")

# base_results_dir = os.path.join(os.getcwd(), "results")
base_results_dir = "results"

run_results_dir = os.path.join(
    base_results_dir,
    f"{timestamp}_fourier_unet3d_local_integral",
)

os.makedirs(run_results_dir, exist_ok=True)

print(f"Results will be saved to: {run_results_dir}\n")


# =========================================================
# Utility
# =========================================================
def skip_data(data: np.ndarray, param: int) -> np.ndarray:
    if param < 0:
        raise ValueError("param must be non-negative")

    return data[::param + 1]


# =========================================================
# Hyperparameters
# =========================================================
window_size = 10
window_stride = 1
skip_param = 3

batch_size = 16

num_epochs = 500
max_frames = 300

files_per_epoch = 76
windows_per_file = None

shuffle_files_each_cycle = True
shuffle_windows = True
file_shuffle_seed = 42

# Loss only on centered inner area of 121x121 crop, ful res used
loss_crop_size = 121

# U-Net model width.
base_channels = 16

# ---------------------------------------------------------
# Learned local-integral resampling settings
# ---------------------------------------------------------
# These are the fixed internal spatial grids.
# Training:
#     121 -> 60 -> 30 -> 15
#
# High-resolution inference can later use:
#     301 -> 60 -> 30 -> 15
#
# The decoder reverses the hierarchy:
#     15 -> 30 -> 60 -> native H x W
#
# Coordinates are normalized to [0,1] across the physical domain.
# Therefore radius=0.08 means 8% of the domain width/height.
integral_grid_1 = 60
integral_grid_2 = 30
integral_grid_3 = 15

integral_num_basis = 8
integral_radius = 0.08

learning_rate = 1e-3


# =========================================================
# File list
# =========================================================

folder_path = os.path.join(
    os.getcwd(),
    "Train_data",
)

data_files = sorted(
    glob.glob(
        os.path.join(folder_path, "*_data.npy")
    )
)

if len(data_files) == 0:
    raise FileNotFoundError(
        f"No *_data.npy files found in {folder_path}"
    )


def paired_files(data_path):
    return {
        "velocity": data_path.replace(
            "_data.npy",
            "_velocity.npy",
        ),
        "pulse_x": data_path.replace(
            "_data.npy",
            "_pulse_xcoord.npy",
        ),
        "pulse_y": data_path.replace(
            "_data.npy",
            "_pulse_ycoord.npy",
        ),
        "xcoord": data_path.replace(
            "_data.npy",
            "_xcoord.npy",
        ),
        "ycoord": data_path.replace(
            "_data.npy",
            "_ycoord.npy",
        ),
        "tcoord": data_path.replace(
            "_data.npy",
            "_tcoord.npy",
        ),
    }


for data_path in data_files:
    for key, path in paired_files(data_path).items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing {key} file: {path}"
            )


num_files = len(data_files)


# =========================================================
# Handle files_per_epoch=None safely
# =========================================================
# files_per_epoch = None:
#     use all files in every epoch
#
# windows_per_file = None:
#     use all possible temporal windows from each file
# =========================================================
if files_per_epoch is None:
    files_per_epoch_effective = num_files
    files_per_epoch_print = "ALL FILES"
    cycle_epochs = 1
else:
    files_per_epoch_effective = files_per_epoch
    files_per_epoch_print = files_per_epoch

    cycle_epochs = math.ceil(
        num_files / files_per_epoch_effective
    )


effective_frames = len(
    np.arange(
        0,
        max_frames,
        skip_param + 1,
    )
)


if windows_per_file is None:
    windows_per_file_print = "ALL POSSIBLE WINDOWS"

    pairs_per_file = max(
        0,
        effective_frames - 2 * window_size + 1,
    )

    expected_pairs = (
        files_per_epoch_effective * pairs_per_file
    )
else:
    windows_per_file_print = windows_per_file

    expected_pairs = (
        files_per_epoch_effective
        * windows_per_file
    )


print(f"Found {num_files} data files.")
print(
    f"Using first {max_frames} frames "
    f"from each file."
)
print(f"Files per epoch: {files_per_epoch_print}")
print(
    f"Windows per file: "
    f"{windows_per_file_print}"
)
print(
    f"Expected training pairs per epoch: "
    f"about {expected_pairs}"
)
print(
    f"One full file-coverage cycle: "
    f"{cycle_epochs} epochs"
)
print(
    f"Loss computed on centered "
    f"{loss_crop_size}x{loss_crop_size} region\n"
)


# =========================================================
# Dataset
# =========================================================
class Wave3DIterable(IterableDataset):
    def __init__(
        self,
        data_files,
        window_size,
        skip_param,
        window_stride,
        max_frames=200,
        files_per_epoch=None,
        windows_per_file=None,
        shuffle_files_each_cycle=True,
        shuffle_windows=True,
        file_shuffle_seed=42,
    ):
        self.data_files = data_files
        self.window_size = window_size
        self.skip_param = skip_param
        self.window_stride = window_stride
        self.max_frames = max_frames
        self.files_per_epoch = files_per_epoch
        self.windows_per_file = windows_per_file
        self.shuffle_files_each_cycle = (
            shuffle_files_each_cycle
        )
        self.shuffle_windows = shuffle_windows
        self.file_shuffle_seed = file_shuffle_seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def get_epoch_file_indices(self):
        n_files = len(self.data_files)

        if self.files_per_epoch is None:
            return np.arange(n_files)

        cycle_epochs = math.ceil(
            n_files / self.files_per_epoch
        )

        cycle_id = self.epoch // cycle_epochs
        epoch_in_cycle = self.epoch % cycle_epochs

        if self.shuffle_files_each_cycle:
            rng = np.random.default_rng(
                self.file_shuffle_seed + cycle_id
            )

            file_order = rng.permutation(n_files)
        else:
            file_order = np.arange(n_files)

        start = (
            epoch_in_cycle
            * self.files_per_epoch
        )

        end = min(
            start + self.files_per_epoch,
            n_files,
        )

        return file_order[start:end]

    def __iter__(self):
        file_indices = self.get_epoch_file_indices()

        print(
            f"Epoch {self.epoch + 1}: using "
            f"{len(file_indices)} files "
            f"from rotating file schedule",
            flush=True,
        )

        for file_idx in file_indices:
            data_path = self.data_files[file_idx]
            files = paired_files(data_path)

            sim_raw = np.load(
                data_path,
                mmap_mode="r",
            ).astype(np.float32)

            if sim_raw.ndim != 3:
                raise ValueError(
                    "Expected data shape (x,y,t), "
                    f"got {sim_raw.shape}"
                )

            # Saved shape:
            #     (x, y, t)
            #
            # Training shape:
            #     (t, x, y)
            sim_raw = np.transpose(
                sim_raw,
                (2, 0, 1),
            )

            sim_raw = sim_raw[:self.max_frames]

            T, H, W = sim_raw.shape

            velocity2d = np.load(
                files["velocity"]
            ).astype(np.float32)

            pulse_x2d = np.load(
                files["pulse_x"]
            ).astype(np.float32)

            pulse_y2d = np.load(
                files["pulse_y"]
            ).astype(np.float32)

            xcoord2d = np.load(
                files["xcoord"]
            ).astype(np.float32)

            ycoord2d = np.load(
                files["ycoord"]
            ).astype(np.float32)

            for name, arr in [
                ("velocity", velocity2d),
                ("pulse_x", pulse_x2d),
                ("pulse_y", pulse_y2d),
                ("xcoord", xcoord2d),
                ("ycoord", ycoord2d),
            ]:
                if arr.shape != (H, W):
                    raise ValueError(
                        f"{name} shape {arr.shape} "
                        f"does not match data {(H, W)}"
                    )

            tcoord_raw = np.load(
                files["tcoord"]
            ).astype(np.float32)

            if tcoord_raw.ndim == 3:
                tcoord = np.transpose(
                    tcoord_raw,
                    (2, 0, 1),
                )

                tcoord = tcoord[:self.max_frames]

            elif tcoord_raw.ndim == 2:
                tcoord = np.tile(
                    tcoord_raw[None, :, :],
                    (T, 1, 1),
                )

            else:
                raise ValueError(
                    f"Invalid tcoord shape: "
                    f"{tcoord_raw.shape}"
                )

            if tcoord.shape != sim_raw.shape:
                raise ValueError(
                    f"tcoord shape {tcoord.shape} "
                    f"does not match data "
                    f"{sim_raw.shape}"
                )

            velocity = np.tile(
                velocity2d[None, :, :],
                (T, 1, 1),
            )

            pulse_x = np.tile(
                pulse_x2d[None, :, :],
                (T, 1, 1),
            )

            pulse_y = np.tile(
                pulse_y2d[None, :, :],
                (T, 1, 1),
            )

            xcoord = np.tile(
                xcoord2d[None, :, :],
                (T, 1, 1),
            )

            ycoord = np.tile(
                ycoord2d[None, :, :],
                (T, 1, 1),
            )

            sim = skip_data(
                sim_raw,
                self.skip_param,
            ).astype(np.float32)

            velocity = skip_data(
                velocity,
                self.skip_param,
            ).astype(np.float32)

            pulse_x = skip_data(
                pulse_x,
                self.skip_param,
            ).astype(np.float32)

            pulse_y = skip_data(
                pulse_y,
                self.skip_param,
            ).astype(np.float32)

            xcoord = skip_data(
                xcoord,
                self.skip_param,
            ).astype(np.float32)

            ycoord = skip_data(
                ycoord,
                self.skip_param,
            ).astype(np.float32)

            tcoord = skip_data(
                tcoord,
                self.skip_param,
            ).astype(np.float32)

            T = sim.shape[0]

            max_i = (
                T
                - 2 * self.window_size
                + 1
            )

            if max_i <= 0:
                continue

            window_indices = np.arange(
                0,
                max_i,
                self.window_stride,
            )

            if self.shuffle_windows:
                np.random.shuffle(window_indices)

            if self.windows_per_file is not None:
                window_indices = window_indices[
                    :self.windows_per_file
                ]

            for i in window_indices:
                x_sim = sim[
                    i : i + self.window_size
                ]

                y_sim = sim[
                    i + self.window_size :
                    i + 2 * self.window_size
                ]

                x = np.stack(
                    [
                        x_sim,
                        velocity[
                            i :
                            i + self.window_size
                        ],
                        pulse_x[
                            i :
                            i + self.window_size
                        ],
                        pulse_y[
                            i :
                            i + self.window_size
                        ],
                        xcoord[
                            i :
                            i + self.window_size
                        ],
                        ycoord[
                            i :
                            i + self.window_size
                        ],
                        tcoord[
                            i :
                            i + self.window_size
                        ],
                    ],
                    axis=0,
                )
                # Shape:
                # (7, window_size, H, W)

                y = np.stack(
                    [y_sim],
                    axis=0,
                )
                # Shape:
                # (1, window_size, H, W)

                yield (
                    torch.from_numpy(x).float(),
                    torch.from_numpy(y).float(),
                )


dataset = Wave3DIterable(
    data_files=data_files,
    window_size=window_size,
    skip_param=skip_param,
    window_stride=window_stride,
    max_frames=max_frames,
    files_per_epoch=files_per_epoch,
    windows_per_file=windows_per_file,
    shuffle_files_each_cycle=(
        shuffle_files_each_cycle
    ),
    shuffle_windows=shuffle_windows,
    file_shuffle_seed=file_shuffle_seed,
)


dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)


print("Dataset and DataLoader prepared.\n")


# =========================================================
# Fourier U-Net building blocks
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
# Model
# This is equivalent to use 48 modes, (24 negative freq, 24 positive freq)
modes_t=24
modes_x=24
modes_y=24
operator = FourierUNet3D(
    in_channels=7,
    out_channels=1,
    base_channels=base_channels,
    modes_t=modes_t,
    modes_x=modes_x,
    modes_y=modes_y,
    grid1=integral_grid_1,
    grid2=integral_grid_2,
    grid3=integral_grid_3,
    integral_radius=integral_radius,
    integral_num_basis=integral_num_basis,
).to(device)


criterion = nn.MSELoss()


optimizer = optim.Adam(
    operator.parameters(),
    lr=learning_rate,
)


scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=50,
    gamma=0.5,
)


num_parameters = sum(
    parameter.numel()
    for parameter in operator.parameters()
    if parameter.requires_grad
)


print("Fourier 3D U-Net with local-integral resampling initialized.")
print(
    f"Trainable parameters: "
    f"{num_parameters:,}\n"
)

print(
    "Local-integral spatial hierarchy: "
    f"native -> {integral_grid_1} -> "
    f"{integral_grid_2} -> {integral_grid_3} -> "
    f"{integral_grid_2} -> {integral_grid_1} -> native"
)
print(
    f"Local-integral radius (normalized physical domain): "
    f"{integral_radius}"
)
print(
    f"Local-integral basis functions: "
    f"{integral_num_basis}\n"
)


# =========================================================
# Optional shape test
# =========================================================
# This verifies that the model preserves the input
# temporal and spatial dimensions.
with torch.no_grad():
    test_input = torch.zeros(
        1,
        7,
        window_size,
        121,
        121,
        device=device,
    )

    test_output = operator(test_input)

    expected_shape = (
        1,
        1,
        window_size,
        121,
        121,
    )

    if test_output.shape != expected_shape:
        raise RuntimeError(
            "Unexpected U-Net output shape. "
            f"Expected {expected_shape}, "
            f"got {tuple(test_output.shape)}"
        )

    print(
        "Shape test passed: "
        f"{tuple(test_input.shape)} -> "
        f"{tuple(test_output.shape)}\n"
    )

del test_input
del test_output

if torch.cuda.is_available():
    torch.cuda.empty_cache()


# =========================================================
# Training
# =========================================================
train_losses = []

print("Starting training...\n")

total_start = time.time()


for epoch in range(1, num_epochs + 1):
    dataset.set_epoch(epoch - 1)

    epoch_start = time.time()
    operator.train()

    running_loss = 0.0
    batch_count = 0

    for inputs, targets in dataloader:
        batch_count += 1

        inputs = inputs.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        outputs = operator(inputs)

        # Verify that model and target shapes match.
        if outputs.shape != targets.shape:
            raise RuntimeError(
                "Output and target shapes differ: "
                f"outputs={tuple(outputs.shape)}, "
                f"targets={tuple(targets.shape)}"
            )

        # =================================================
        # Loss only on centered inner spatial area
        #
        # outputs/targets:
        # (batch, 1, time, x, y)
        # =================================================
        _, _, _, H, W = outputs.shape

        if (
            loss_crop_size > H
            or loss_crop_size > W
        ):
            raise ValueError(
                f"loss_crop_size={loss_crop_size} "
                f"is larger than output size "
                f"{(H, W)}"
            )

        margin_x = (
            H - loss_crop_size
        ) // 2

        margin_y = (
            W - loss_crop_size
        ) // 2

        x_start = margin_x
        x_end = margin_x + loss_crop_size

        y_start = margin_y
        y_end = margin_y + loss_crop_size

        outputs_center = outputs[
            :,
            :,
            :,
            x_start:x_end,
            y_start:y_end,
        ]

        targets_center = targets[
            :,
            :,
            :,
            x_start:x_end,
            y_start:y_end,
        ]

        loss = criterion(
            outputs_center,
            targets_center,
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    if batch_count == 0:
        raise RuntimeError(
            "No batches were produced."
        )

    epoch_loss = (
        running_loss / batch_count
    )

    train_losses.append(epoch_loss)

    scheduler.step()

    epoch_time = (
        time.time() - epoch_start
    )

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch {epoch:4d}/{num_epochs} | "
        f"Loss: {epoch_loss:.6e} | "
        f"Batches: {batch_count} | "
        f"LR: {current_lr:.3e} | "
        f"Time: {epoch_time:.2f}s",
        flush=True,
    )


total_time = time.time() - total_start

print(
    f"\nTraining completed in "
    f"{total_time / 60:.2f} minutes."
)


# =========================================================
# Save outputs
# =========================================================
model_path = os.path.join(
    run_results_dir,
    "fourier_unet3d_local_integral_model.pth",
)

torch.save(
    operator.state_dict(),
    model_path,
)

print(f"Model saved to {model_path}")


# Save model configuration for testing.
config_path = os.path.join(
    run_results_dir,
    "fourier_unet3d_local_integral_config.npy",
)

np.save(
    config_path,
    {
        "in_channels": 7,
        "out_channels": 1,
        "base_channels": base_channels,
        "channel_widths": [base_channels, 24, 32, 48],
        "modes_t": modes_t,
        "modes_x": modes_x,
        "modes_y": modes_y,
        "window_size": window_size,
        "skip_param": skip_param,
        "loss_crop_size": loss_crop_size,
        "integral_grid_1": integral_grid_1,
        "integral_grid_2": integral_grid_2,
        "integral_grid_3": integral_grid_3,
        "integral_radius": integral_radius,
        "integral_num_basis": integral_num_basis,
        "resampling_type": "separable_local_integral_DISCO_inspired_no_L1_normalization",
    },
    allow_pickle=True,
)

print(
    f"Model configuration saved to "
    f"{config_path}"
)


# Save training loss plot.
loss_plot_path = os.path.join(
    run_results_dir,
    "training_loss.png",
)

plt.figure()

plt.plot(
    range(
        1,
        len(train_losses) + 1,
    ),
    train_losses,
)

plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Fourier 3D U-Net + Local Integral Resampling Training Loss")
plt.tight_layout()
plt.savefig(
    loss_plot_path,
    dpi=200,
)
plt.close()


# Save numerical loss values.
loss_values_path = os.path.join(
    run_results_dir,
    "train_losses.npy",
)

np.save(
    loss_values_path,
    np.array(
        train_losses,
        dtype=np.float32,
    ),
)


print(
    f"Training loss plot saved to "
    f"{loss_plot_path}"
)

print(
    f"Training losses saved to "
    f"{loss_values_path}"
)
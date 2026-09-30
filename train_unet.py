import os
import glob
import time
import math
import datetime

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
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

base_results_dir = os.path.join(os.getcwd(), "results")
run_results_dir = os.path.join(
    base_results_dir,
    f"{timestamp}_unet3d",
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
# 3D U-Net building blocks
# =========================================================
class DoubleConv3D(nn.Module):
    """
    Two consecutive 3D convolutions.

    GroupNorm is used instead of BatchNorm because
    3D models often require small batch sizes.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super().__init__()

        # Choose a valid number of groups.
        if out_channels >= 8:
            num_groups = 8
        elif out_channels >= 4:
            num_groups = 4
        else:
            num_groups = 1

        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=num_groups,
                num_channels=out_channels,
            ),
            nn.GELU(),

            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=num_groups,
                num_channels=out_channels,
            ),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class UNet3D(nn.Module):
    """
    3D U-Net for block-to-block wave prediction.

    Input:
        (B, 7, T, H, W)

    Output:
        (B, 1, T, H, W)

    Time is preserved during pooling. Only H and W
    are spatially downsampled.
    """

    def __init__(
        self,
        in_channels=7,
        out_channels=1,
        base_channels=16,
    ):
        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        # -------------------------
        # Encoder
        # -------------------------
        self.encoder1 = DoubleConv3D(
            in_channels,
            c1,
        )

        self.encoder2 = DoubleConv3D(
            c1,
            c2,
        )

        self.encoder3 = DoubleConv3D(
            c2,
            c3,
        )

        self.bottleneck = DoubleConv3D(
            c3,
            c4,
        )

        # Pool only along spatial dimensions.
        self.pool = nn.MaxPool3d(
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2),
        )

        # -------------------------
        # Decoder
        # -------------------------
        self.decoder3 = DoubleConv3D(
            c4 + c3,
            c3,
        )

        self.decoder2 = DoubleConv3D(
            c3 + c2,
            c2,
        )

        self.decoder1 = DoubleConv3D(
            c2 + c1,
            c1,
        )

        self.output_layer = nn.Conv3d(
            c1,
            out_channels,
            kernel_size=1,
        )

    @staticmethod
    def resize_to_skip(x, skip):
        """
        Resize decoder features to exactly match the
        temporal and spatial dimensions of skip features.

        This safely handles odd dimensions such as:
            121 x 121
            301 x 301
        """

        return F.interpolate(
            x,
            size=skip.shape[2:],
            mode="trilinear",
            align_corners=False,
        )

    def forward(self, x):
        # -------------------------
        # Encoder
        # -------------------------
        enc1 = self.encoder1(x)
        # (B, c1, T, H, W)

        enc2 = self.encoder2(
            self.pool(enc1)
        )
        # (B, c2, T, H/2, W/2)

        enc3 = self.encoder3(
            self.pool(enc2)
        )
        # (B, c3, T, H/4, W/4)

        bottleneck = self.bottleneck(
            self.pool(enc3)
        )
        # (B, c4, T, H/8, W/8)

        # -------------------------
        # Decoder level 3
        # -------------------------
        dec3 = self.resize_to_skip(
            bottleneck,
            enc3,
        )

        dec3 = torch.cat(
            [dec3, enc3],
            dim=1,
        )

        dec3 = self.decoder3(dec3)

        # -------------------------
        # Decoder level 2
        # -------------------------
        dec2 = self.resize_to_skip(
            dec3,
            enc2,
        )

        dec2 = torch.cat(
            [dec2, enc2],
            dim=1,
        )

        dec2 = self.decoder2(dec2)

        # -------------------------
        # Decoder level 1
        # -------------------------
        dec1 = self.resize_to_skip(
            dec2,
            enc1,
        )

        dec1 = torch.cat(
            [dec1, enc1],
            dim=1,
        )

        dec1 = self.decoder1(dec1)

        output = self.output_layer(dec1)

        return output


# =========================================================
# Model
# =========================================================
operator = UNet3D(
    in_channels=7,
    out_channels=1,
    base_channels=base_channels,
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


print("3D U-Net initialized.")
print(
    f"Trainable parameters: "
    f"{num_parameters:,}\n"
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
    "unet3d_model.pth",
)

torch.save(
    operator.state_dict(),
    model_path,
)

print(f"Model saved to {model_path}")


# Save model configuration for testing.
config_path = os.path.join(
    run_results_dir,
    "unet3d_config.npy",
)

np.save(
    config_path,
    {
        "in_channels": 7,
        "out_channels": 1,
        "base_channels": base_channels,
        "window_size": window_size,
        "skip_param": skip_param,
        "loss_crop_size": loss_crop_size,
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
plt.title("3D U-Net Training Loss")
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

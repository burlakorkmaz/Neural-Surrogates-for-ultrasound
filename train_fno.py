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
from neuralop.models import FNO
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")


now = datetime.datetime.now()
timestamp = now.strftime("%d%m%Y_%H%M")
base_results_dir = "results"
run_results_dir = os.path.join(base_results_dir, timestamp)
os.makedirs(run_results_dir, exist_ok=True)

print(f"Results will be saved to: {run_results_dir}\n")


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


# =========================================================
# File list
# =========================================================
folder_path = os.path.join(os.getcwd(), "Train_data")

data_files = sorted(glob.glob(os.path.join(folder_path, "*_data.npy")))

if len(data_files) == 0:
    raise FileNotFoundError(f"No *_data.npy files found in {folder_path}")


def paired_files(data_path):
    return {
        "velocity": data_path.replace("_data.npy", "_velocity.npy"),
        "pulse_x": data_path.replace("_data.npy", "_pulse_xcoord.npy"),
        "pulse_y": data_path.replace("_data.npy", "_pulse_ycoord.npy"),
        "xcoord": data_path.replace("_data.npy", "_xcoord.npy"),
        "ycoord": data_path.replace("_data.npy", "_ycoord.npy"),
        "tcoord": data_path.replace("_data.npy", "_tcoord.npy"),
    }


for data_path in data_files:
    for key, path in paired_files(data_path).items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {key} file: {path}")


num_files = len(data_files)

# =========================================================
# Handle files_per_epoch=None safely
# =========================================================
# files_per_epoch = None means:
#     use ALL files in every epoch
#
# windows_per_file = None means:
#     use ALL possible temporal windows from each selected file
# =========================================================
if files_per_epoch is None:
    files_per_epoch_effective = num_files
    files_per_epoch_print = "ALL FILES"
    cycle_epochs = 1
else:
    files_per_epoch_effective = files_per_epoch
    files_per_epoch_print = files_per_epoch
    cycle_epochs = math.ceil(num_files / files_per_epoch_effective)

effective_frames = len(np.arange(0, max_frames, skip_param + 1))

if windows_per_file is None:
    windows_per_file_print = "ALL POSSIBLE WINDOWS"
    pairs_per_file = max(0, effective_frames - 2 * window_size + 1)
    expected_pairs = files_per_epoch_effective * pairs_per_file
else:
    windows_per_file_print = windows_per_file
    expected_pairs = files_per_epoch_effective * windows_per_file

print(f"Found {num_files} cropped data files.")
print(f"Using first {max_frames} frames from each cropped file.")
print(f"Files per epoch: {files_per_epoch_print}")
print(f"Windows per file: {windows_per_file_print}")
print(f"Expected training pairs per epoch: about {expected_pairs}")
print(f"One full file-coverage cycle: {cycle_epochs} epochs")
print(f"Loss computed on centered {loss_crop_size}x{loss_crop_size} region\n")


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
        self.shuffle_files_each_cycle = shuffle_files_each_cycle
        self.shuffle_windows = shuffle_windows
        self.file_shuffle_seed = file_shuffle_seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def get_epoch_file_indices(self):
        n_files = len(self.data_files)

        if self.files_per_epoch is None:
            return np.arange(n_files)

        cycle_epochs = math.ceil(n_files / self.files_per_epoch)
        cycle_id = self.epoch // cycle_epochs
        epoch_in_cycle = self.epoch % cycle_epochs

        if self.shuffle_files_each_cycle:
            rng = np.random.default_rng(self.file_shuffle_seed + cycle_id)
            file_order = rng.permutation(n_files)
        else:
            file_order = np.arange(n_files)

        start = epoch_in_cycle * self.files_per_epoch
        end = min(start + self.files_per_epoch, n_files)

        return file_order[start:end]

    def __iter__(self):
        file_indices = self.get_epoch_file_indices()

        print(
            f"Epoch {self.epoch + 1}: using {len(file_indices)} files "
            f"from rotating file schedule",
            flush=True,
        )

        for file_idx in file_indices:
            data_path = self.data_files[file_idx]
            files = paired_files(data_path)

            sim_raw = np.load(data_path, mmap_mode="r").astype(np.float32)

            if sim_raw.ndim != 3:
                raise ValueError(f"Expected data shape (x,y,t), got {sim_raw.shape}")

            # saved: (x, y, t) -> training: (t, x, y)
            sim_raw = np.transpose(sim_raw, (2, 0, 1))
            sim_raw = sim_raw[:self.max_frames]

            T, H, W = sim_raw.shape

            velocity2d = np.load(files["velocity"]).astype(np.float32)
            pulse_x2d = np.load(files["pulse_x"]).astype(np.float32)
            pulse_y2d = np.load(files["pulse_y"]).astype(np.float32)
            xcoord2d = np.load(files["xcoord"]).astype(np.float32)
            ycoord2d = np.load(files["ycoord"]).astype(np.float32)

            for name, arr in [
                ("velocity", velocity2d),
                ("pulse_x", pulse_x2d),
                ("pulse_y", pulse_y2d),
                ("xcoord", xcoord2d),
                ("ycoord", ycoord2d),
            ]:
                if arr.shape != (H, W):
                    raise ValueError(
                        f"{name} shape {arr.shape} does not match data {(H, W)}"
                    )

            tcoord_raw = np.load(files["tcoord"]).astype(np.float32)

            if tcoord_raw.ndim == 3:
                tcoord = np.transpose(tcoord_raw, (2, 0, 1))
                tcoord = tcoord[:self.max_frames]
            elif tcoord_raw.ndim == 2:
                tcoord = np.tile(tcoord_raw[None, :, :], (T, 1, 1))
            else:
                raise ValueError(f"Invalid tcoord shape: {tcoord_raw.shape}")

            if tcoord.shape != sim_raw.shape:
                raise ValueError(
                    f"tcoord shape {tcoord.shape} does not match data {sim_raw.shape}"
                )

            velocity = np.tile(velocity2d[None, :, :], (T, 1, 1))
            pulse_x = np.tile(pulse_x2d[None, :, :], (T, 1, 1))
            pulse_y = np.tile(pulse_y2d[None, :, :], (T, 1, 1))
            xcoord = np.tile(xcoord2d[None, :, :], (T, 1, 1))
            ycoord = np.tile(ycoord2d[None, :, :], (T, 1, 1))

            sim = skip_data(sim_raw, self.skip_param).astype(np.float32)
            velocity = skip_data(velocity, self.skip_param).astype(np.float32)
            pulse_x = skip_data(pulse_x, self.skip_param).astype(np.float32)
            pulse_y = skip_data(pulse_y, self.skip_param).astype(np.float32)
            xcoord = skip_data(xcoord, self.skip_param).astype(np.float32)
            ycoord = skip_data(ycoord, self.skip_param).astype(np.float32)
            tcoord = skip_data(tcoord, self.skip_param).astype(np.float32)

            T = sim.shape[0]
            max_i = T - 2 * self.window_size + 1

            if max_i <= 0:
                continue

            window_indices = np.arange(0, max_i, self.window_stride)

            if self.shuffle_windows:
                np.random.shuffle(window_indices)

            if self.windows_per_file is not None:
                window_indices = window_indices[:self.windows_per_file]

            for i in window_indices:
                x_sim = sim[i : i + self.window_size]
                y_sim = sim[i + self.window_size : i + 2 * self.window_size]

                x = np.stack(
                    [
                        x_sim,
                        velocity[i : i + self.window_size],
                        pulse_x[i : i + self.window_size],
                        pulse_y[i : i + self.window_size],
                        xcoord[i : i + self.window_size],
                        ycoord[i : i + self.window_size],
                        tcoord[i : i + self.window_size],
                    ],
                    axis=0,
                )  # (7, window, x, y)

                y = np.stack([y_sim], axis=0)  # (1, window, x, y)

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
    shuffle_files_each_cycle=shuffle_files_each_cycle,
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
# Model
# =========================================================
modes_t, modes_x, modes_y = 48, 48, 48

operator = FNO(
    n_modes=(modes_t, modes_x, modes_y),
    in_channels=7,
    out_channels=1,
    hidden_channels=64,
    n_layers=4,
).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(operator.parameters(), lr=1e-3)

scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=50,
    gamma=0.5,
)

print("Model initialized.\n")


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

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = operator(inputs)

        # =================================================
        # Loss only on centered inner spatial area
        # outputs/targets shape: (batch, 1, time, x, y)
        # =================================================
        _, _, _, H, W = outputs.shape

        if loss_crop_size > H or loss_crop_size > W:
            raise ValueError(
                f"loss_crop_size={loss_crop_size} is larger than output size {(H, W)}"
            )

        margin_x = (H - loss_crop_size) // 2
        margin_y = (W - loss_crop_size) // 2

        x_start = margin_x
        x_end = margin_x + loss_crop_size
        y_start = margin_y
        y_end = margin_y + loss_crop_size

        outputs_center = outputs[
            :, :, :,
            x_start:x_end,
            y_start:y_end,
        ]

        targets_center = targets[
            :, :, :,
            x_start:x_end,
            y_start:y_end,
        ]

        loss = criterion(outputs_center, targets_center)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    if batch_count == 0:
        raise RuntimeError("No batches were produced.")

    epoch_loss = running_loss / batch_count
    train_losses.append(epoch_loss)

    scheduler.step()

    epoch_time = time.time() - epoch_start

    print(
        f"Epoch {epoch:4d}/{num_epochs} | "
        f"Loss: {epoch_loss:.6e} | "
        f"Batches: {batch_count} | "
        f"Time: {epoch_time:.2f}s",
        flush=True,
    )


total_time = time.time() - total_start

print(f"\nTraining completed in {total_time / 60:.2f} minutes.")


# =========================================================
# Save outputs
# =========================================================
model_path = os.path.join(run_results_dir, "fno3d_model.pth")
torch.save(operator.state_dict(), model_path)

print(f"Model saved to {model_path}")

loss_plot_path = os.path.join(run_results_dir, "training_loss.png")

plt.figure()
plt.plot(range(1, len(train_losses) + 1), train_losses)
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss Over Epochs")
plt.tight_layout()
plt.savefig(loss_plot_path)
plt.close()

np.save(
    os.path.join(run_results_dir, "train_losses.npy"),
    np.array(train_losses),
)

print(f"Training loss plot saved to {loss_plot_path}")
print("Training losses saved.")
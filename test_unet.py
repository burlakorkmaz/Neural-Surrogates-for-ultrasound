import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

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
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# =========================================================
# Common test settings
# =========================================================
MAX_FRAMES = 300
BASE_CHANNELS = 16

# These were the original-frame plots.
ORIGINAL_FRAMES_TO_PLOT = [
    40, 50, 60, 70, 80, 90, 100, 110, 120, 150, 180, 296
]

# Run the SAME trained U-Net at both resolutions.
TEST_CONFIGS = {
    "high_resolution": {
        "window_frames": 10,
        "temporal_skip_param": 3,
        "target_spatial_size": (301, 301),
        "spatial_start_mode": "even",
        "root_output_directory": "test_results_unet3d_high_resolution_301",
        "summary_file": "high_resolution_relative_squared_l2.txt",
    },
    "low_resolution": {
        "window_frames": 10,
        "temporal_skip_param": 3,
        "target_spatial_size": (121, 121),
        "spatial_start_mode": "even",
        "root_output_directory": "test_results_unet3d_low_resolution_121",
        "summary_file": "low_resolution_relative_squared_l2.txt",
    },
}


# =========================================================
# General helpers
# =========================================================
def skip_data(
    data: np.ndarray,
    param: int,
) -> np.ndarray:
    if param < 0:
        raise ValueError(
            "param must be non-negative"
        )

    return data[::param + 1]


def separate_symmetric_limits(arr):
    vmax = float(np.max(np.abs(arr)))

    if vmax == 0 or np.isnan(vmax):
        vmax = 1e-12

    return -vmax, vmax


# =========================================================
# Spatial downsampling
# Same rule as the original test code
# =========================================================
def make_downsample_indices(
    n_original,
    n_target,
    start_mode,
):
    if n_target > n_original:
        raise ValueError(
            f"Target size {n_target} cannot be larger "
            f"than original size {n_original}"
        )

    if start_mode == "even":
        start = 0
    elif start_mode == "odd":
        start = 1
    else:
        raise ValueError(
            "start_mode must be 'even' or 'odd'"
        )

    indices = np.linspace(
        start,
        n_original - 1,
        n_target,
    )

    indices = np.round(
        indices
    ).astype(int)

    indices = np.clip(
        indices,
        0,
        n_original - 1,
    )

    if len(np.unique(indices)) != len(indices):
        raise ValueError(
            "Duplicate indices produced for "
            f"n_original={n_original}, "
            f"n_target={n_target}, "
            f"start_mode={start_mode}"
        )

    return indices


def spatial_downsample_2d(
    arr,
    target_size,
    start_mode,
):
    if arr.ndim != 2:
        raise ValueError(
            f"Expected 2D array, got {arr.shape}"
        )

    target_x, target_y = target_size

    # If already at target resolution, keep unchanged.
    if arr.shape == (target_x, target_y):
        return arr

    x_indices = make_downsample_indices(
        arr.shape[0],
        target_x,
        start_mode,
    )

    y_indices = make_downsample_indices(
        arr.shape[1],
        target_y,
        start_mode,
    )

    return arr[
        np.ix_(
            x_indices,
            y_indices,
        )
    ]


def spatial_downsample_3d_xyt(
    arr,
    target_size,
    start_mode,
):
    """
    Input:
        arr shape = (x, y, time)

    Output:
        shape = (target_x, target_y, time)

    Time is not downsampled here.
    """
    if arr.ndim != 3:
        raise ValueError(
            f"Expected 3D array, got {arr.shape}"
        )

    target_x, target_y = target_size

    # If already at target spatial resolution, keep unchanged.
    if arr.shape[:2] == (target_x, target_y):
        return arr

    x_indices = make_downsample_indices(
        arr.shape[0],
        target_x,
        start_mode,
    )

    y_indices = make_downsample_indices(
        arr.shape[1],
        target_y,
        start_mode,
    )

    time_indices = np.arange(arr.shape[2])

    return arr[
        np.ix_(
            x_indices,
            y_indices,
            time_indices,
        )
    ]


def downsample_spatial_array(
    arr,
    target_size,
    start_mode,
):
    if arr.ndim == 2:
        return spatial_downsample_2d(
            arr,
            target_size,
            start_mode,
        )

    if arr.ndim == 3:
        return spatial_downsample_3d_xyt(
            arr,
            target_size,
            start_mode,
        )

    raise ValueError(
        f"Expected a 2D or 3D array, got {arr.shape}"
    )


# =========================================================
# File discovery
# =========================================================
def get_test_files(data_folder):
    simulation_files = sorted(
        glob.glob(
            os.path.join(
                data_folder,
                "wave_source_*_xyt_kpa.npy",
            )
        )
    )

    if len(simulation_files) == 0:
        raise FileNotFoundError(
            f"No simulation files found in {data_folder}"
        )

    xcoord_file = os.path.join(
        data_folder,
        "X_coordinates_mm.npy",
    )

    ycoord_file = os.path.join(
        data_folder,
        "Y_coordinates_mm.npy",
    )

    tcoord_file = os.path.join(
        data_folder,
        "T_coordinates_us.npy",
    )

    for path in [
        xcoord_file,
        ycoord_file,
        tcoord_file,
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing required file: {path}"
            )

    return (
        simulation_files,
        xcoord_file,
        ycoord_file,
        tcoord_file,
    )


# =========================================================
# Source-specific files
# =========================================================
def source_index_from_sim(sim_file):
    basename = os.path.basename(sim_file)

    match = re.search(
        r"wave_source_(\d+)_xyt_kpa\.npy",
        basename,
    )

    if match is None:
        raise ValueError(
            f"Could not extract source index from {basename}"
        )

    return match.group(1)


def pulse_files_for_sim(sim_file):
    source_index = source_index_from_sim(sim_file)
    folder = os.path.dirname(sim_file)

    pulse_x_file = os.path.join(
        folder,
        f"pulse_source_{source_index}_xcoord.npy",
    )

    pulse_y_file = os.path.join(
        folder,
        f"pulse_source_{source_index}_ycoord.npy",
    )

    if not os.path.exists(pulse_x_file):
        raise FileNotFoundError(
            f"Missing pulse x file: {pulse_x_file}"
        )

    if not os.path.exists(pulse_y_file):
        raise FileNotFoundError(
            f"Missing pulse y file: {pulse_y_file}"
        )

    return pulse_x_file, pulse_y_file


def velocity_file_for_sim(sim_file):
    source_index = source_index_from_sim(sim_file)
    folder = os.path.dirname(sim_file)

    source_velocity_file = os.path.join(
        folder,
        f"velocity_map_source_{source_index}.npy",
    )

    global_velocity_file = os.path.join(
        folder,
        "velocity_map.npy",
    )

    if os.path.exists(source_velocity_file):
        return source_velocity_file

    if os.path.exists(global_velocity_file):
        print(
            f"Warning: {source_velocity_file} not found. "
            f"Using {global_velocity_file}"
        )

        return global_velocity_file

    raise FileNotFoundError(
        "Missing velocity map. Expected either:\n"
        f"  {source_velocity_file}\n"
        "or\n"
        f"  {global_velocity_file}"
    )


# =========================================================
# 3D U-Net architecture
# Must remain identical to training
# =========================================================
class DoubleConv3D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super().__init__()

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

        # Preserve time and downsample only x and y.
        self.pool = nn.MaxPool3d(
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2),
        )

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
        return F.interpolate(
            x,
            size=skip.shape[2:],
            mode="trilinear",
            align_corners=False,
        )

    def forward(self, x):
        enc1 = self.encoder1(x)

        enc2 = self.encoder2(
            self.pool(enc1)
        )

        enc3 = self.encoder3(
            self.pool(enc2)
        )

        bottleneck = self.bottleneck(
            self.pool(enc3)
        )

        dec3 = self.resize_to_skip(
            bottleneck,
            enc3,
        )

        dec3 = torch.cat(
            [dec3, enc3],
            dim=1,
        )

        dec3 = self.decoder3(dec3)

        dec2 = self.resize_to_skip(
            dec3,
            enc2,
        )

        dec2 = torch.cat(
            [dec2, enc2],
            dim=1,
        )

        dec2 = self.decoder2(dec2)

        dec1 = self.resize_to_skip(
            dec2,
            enc1,
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
def load_model(model_file):
    model = UNet3D(
        in_channels=7,
        out_channels=1,
        base_channels=BASE_CHANNELS,
    ).to(device)

    state = torch.load(
        model_file,
        map_location=device,
        weights_only=False,
    )

    if (
        isinstance(state, dict)
        and "model_state_dict" in state
    ):
        state = state["model_state_dict"]

    model.load_state_dict(state)
    model.eval()

    print(
        f"Loaded U-Net model from {model_file}"
    )

    return model


# =========================================================
# Prepare time coordinate
# =========================================================
def prepare_tcoord_xyt(
    tcoord_raw,
    original_shape,
):
    """
    Returns tcoord with shape:
        (x, y, time)
    """

    H, W, T = original_shape

    if tcoord_raw.ndim == 3:
        if tcoord_raw.shape != original_shape:
            raise ValueError(
                f"3D tcoord shape {tcoord_raw.shape} "
                f"does not match simulation {original_shape}"
            )

        return tcoord_raw

    if tcoord_raw.ndim == 2:
        if tcoord_raw.shape != (H, W):
            raise ValueError(
                f"2D tcoord shape {tcoord_raw.shape} "
                f"does not match spatial shape {(H, W)}"
            )

        return np.tile(
            tcoord_raw[:, :, None],
            (1, 1, T),
        )

    raise ValueError(
        f"Invalid tcoord shape: {tcoord_raw.shape}"
    )


# =========================================================
# Prepare one resolution
# =========================================================
def prepare_resolution_data(
    sim_file,
    xcoord_file,
    ycoord_file,
    tcoord_file,
    config,
):
    sim_full_xyt = np.load(
        sim_file
    ).astype(np.float32)

    if sim_full_xyt.ndim != 3:
        raise ValueError(
            "Expected simulation shape (x,y,t), "
            f"got {sim_full_xyt.shape}"
        )

    original_H, original_W, original_T = (
        sim_full_xyt.shape
    )

    original_T_used = min(
        MAX_FRAMES,
        original_T,
    )

    sim_full_xyt = sim_full_xyt[
        :,
        :,
        :original_T_used,
    ]

    velocity_file = velocity_file_for_sim(
        sim_file
    )

    pulse_x_file, pulse_y_file = (
        pulse_files_for_sim(sim_file)
    )

    velocity_full = np.load(
        velocity_file
    ).astype(np.float32)

    pulse_x_full = np.load(
        pulse_x_file
    ).astype(np.float32)

    pulse_y_full = np.load(
        pulse_y_file
    ).astype(np.float32)

    xcoord_full = np.load(
        xcoord_file
    ).astype(np.float32)

    ycoord_full = np.load(
        ycoord_file
    ).astype(np.float32)

    tcoord_raw = np.load(
        tcoord_file
    ).astype(np.float32)

    for name, arr in [
        ("velocity", velocity_full),
        ("pulse_x", pulse_x_full),
        ("pulse_y", pulse_y_full),
        ("xcoord", xcoord_full),
        ("ycoord", ycoord_full),
    ]:
        if arr.shape != (
            original_H,
            original_W,
        ):
            raise ValueError(
                f"{name} shape {arr.shape} does not "
                f"match simulation spatial shape "
                f"{(original_H, original_W)}"
            )

    tcoord_full_xyt = prepare_tcoord_xyt(
        tcoord_raw=tcoord_raw,
        original_shape=(
            original_H,
            original_W,
            original_T,
        ),
    )

    tcoord_full_xyt = tcoord_full_xyt[
        :,
        :,
        :original_T_used,
    ]

    target_size = config["target_spatial_size"]
    start_mode = config["spatial_start_mode"]

    # Apply identical spatial selection to all model inputs.
    sim_ds_xyt = downsample_spatial_array(
        sim_full_xyt,
        target_size,
        start_mode,
    ).astype(np.float32)

    velocity_ds = downsample_spatial_array(
        velocity_full,
        target_size,
        start_mode,
    ).astype(np.float32)

    pulse_x_ds = downsample_spatial_array(
        pulse_x_full,
        target_size,
        start_mode,
    ).astype(np.float32)

    pulse_y_ds = downsample_spatial_array(
        pulse_y_full,
        target_size,
        start_mode,
    ).astype(np.float32)

    xcoord_ds = downsample_spatial_array(
        xcoord_full,
        target_size,
        start_mode,
    ).astype(np.float32)

    ycoord_ds = downsample_spatial_array(
        ycoord_full,
        target_size,
        start_mode,
    ).astype(np.float32)

    tcoord_ds_xyt = downsample_spatial_array(
        tcoord_full_xyt,
        target_size,
        start_mode,
    ).astype(np.float32)

    # (x,y,t) -> (t,x,y)
    sim = np.transpose(
        sim_ds_xyt,
        (2, 0, 1),
    )

    tcoord = np.transpose(
        tcoord_ds_xyt,
        (2, 0, 1),
    )

    temporal_skip_param = config[
        "temporal_skip_param"
    ]

    sim = skip_data(
        sim,
        temporal_skip_param,
    ).astype(np.float32)

    tcoord = skip_data(
        tcoord,
        temporal_skip_param,
    ).astype(np.float32)

    T, H, W = sim.shape

    if (
        H,
        W,
    ) != target_size:
        raise RuntimeError(
            f"Expected spatial size "
            f"{target_size}, got {(H, W)}"
        )

    velocity = np.tile(
        velocity_ds[None, :, :],
        (T, 1, 1),
    )

    pulse_x = np.tile(
        pulse_x_ds[None, :, :],
        (T, 1, 1),
    )

    pulse_y = np.tile(
        pulse_y_ds[None, :, :],
        (T, 1, 1),
    )

    xcoord = np.tile(
        xcoord_ds[None, :, :],
        (T, 1, 1),
    )

    ycoord = np.tile(
        ycoord_ds[None, :, :],
        (T, 1, 1),
    )

    if tcoord.shape != sim.shape:
        raise ValueError(
            f"tcoord shape {tcoord.shape} "
            f"does not match sim shape {sim.shape}"
        )

    represented_original_indices = np.arange(
        0,
        original_T_used,
        temporal_skip_param + 1,
    )[:T]

    return {
        "sim": sim,
        "velocity": velocity,
        "pulse_x": pulse_x,
        "pulse_y": pulse_y,
        "xcoord": xcoord,
        "ycoord": ycoord,
        "tcoord": tcoord,
        "velocity_file": velocity_file,
        "represented_original_indices": represented_original_indices,
    }


# =========================================================
# Autoregressive evaluation
# =========================================================
def evaluate_autoregressive(
    sim_file,
    xcoord_file,
    ycoord_file,
    tcoord_file,
    model,
    config_name,
    config,
):
    sim_name = os.path.splitext(
        os.path.basename(sim_file)
    )[0]

    root_out_dir = config[
        "root_output_directory"
    ]

    save_dir = os.path.join(
        root_out_dir,
        sim_name,
    )

    os.makedirs(
        save_dir,
        exist_ok=True,
    )

    window_frames = config[
        "window_frames"
    ]

    span = window_frames

    print("\n======================================")
    print(
        f"Evaluating: {sim_name} | {config_name}"
    )
    print("======================================")

    prepared = prepare_resolution_data(
        sim_file=sim_file,
        xcoord_file=xcoord_file,
        ycoord_file=ycoord_file,
        tcoord_file=tcoord_file,
        config=config,
    )

    sim = prepared["sim"]
    velocity = prepared["velocity"]
    pulse_x = prepared["pulse_x"]
    pulse_y = prepared["pulse_y"]
    xcoord = prepared["xcoord"]
    ycoord = prepared["ycoord"]
    tcoord = prepared["tcoord"]
    velocity_file = prepared["velocity_file"]
    represented_original_indices = prepared[
        "represented_original_indices"
    ]

    T, H, W = sim.shape

    print(
        "Final model-resolution wave shape:",
        sim.shape,
    )

    print(
        "Original frame indices represented:",
        represented_original_indices,
    )

    if T < 2 * window_frames:
        raise ValueError(
            f"Only {T} effective frames remain, "
            f"but at least {2 * window_frames} are needed."
        )

    # =====================================================
    # Convert to tensors
    # =====================================================
    def to_5d_tensor(arr):
        return (
            torch.from_numpy(arr)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )

    sim_t = to_5d_tensor(sim)
    velocity_t = to_5d_tensor(velocity)
    pulse_x_t = to_5d_tensor(pulse_x)
    pulse_y_t = to_5d_tensor(pulse_y)
    xcoord_t = to_5d_tensor(xcoord)
    ycoord_t = to_5d_tensor(ycoord)
    tcoord_t = to_5d_tensor(tcoord)

    # =====================================================
    # Initial autoregressive input
    # Preserve exact channel order from original code:
    # [wave, velocity, pulse_x, pulse_y, xcoord, ycoord, tcoord]
    # =====================================================
    current_wave = sim_t[
        :,
        :,
        :window_frames,
    ]

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

    expected_input_shape = (
        1,
        7,
        window_frames,
        H,
        W,
    )

    if current_input.shape != expected_input_shape:
        raise RuntimeError(
            f"Expected input shape {expected_input_shape}, "
            f"got {tuple(current_input.shape)}"
        )

    print(
        "Initial input tensor shape:",
        tuple(current_input.shape),
    )

    predictions = [
        sim[:window_frames]
    ]

    current_start = 0
    step = 0

    with torch.inference_mode():
        while (
            current_start + window_frames
        ) < T:

            print(
                f"AR step {step}: "
                f"effective start={current_start}, "
                f"input={tuple(current_input.shape)}",
                flush=True,
            )

            output = model(current_input)

            expected_output_shape = (
                1,
                1,
                window_frames,
                H,
                W,
            )

            if output.shape != expected_output_shape:
                raise RuntimeError(
                    f"Expected output shape "
                    f"{expected_output_shape}, "
                    f"got {tuple(output.shape)}"
                )

            span_effective = min(
                span,
                T - (
                    current_start
                    + window_frames
                ),
            )

            next_prediction = output[
                :,
                :,
                :span_effective,
            ]

            predictions.append(
                next_prediction[
                    0,
                    0,
                ]
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            old_wave = current_input[
                :,
                0:1,
                span_effective:,
            ]

            new_wave = torch.cat(
                [
                    old_wave,
                    next_prediction,
                ],
                dim=2,
            )

            new_start = (
                current_start
                + span_effective
            )

            new_end = (
                new_start
                + window_frames
            )

            def get_window(tensor):
                window = tensor[
                    :,
                    :,
                    new_start:new_end,
                ]

                if (
                    window.shape[2]
                    < window_frames
                ):
                    missing = (
                        window_frames
                        - window.shape[2]
                    )

                    if window.shape[2] == 0:
                        raise RuntimeError(
                            "Cannot pad an empty "
                            "conditioning window."
                        )

                    padding = window[
                        :,
                        :,
                        -1:,
                    ].repeat(
                        1,
                        1,
                        missing,
                        1,
                        1,
                    )

                    window = torch.cat(
                        [
                            window,
                            padding,
                        ],
                        dim=2,
                    )

                return window

            # Preserve same channel order in every AR step.
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

    rollout = np.concatenate(
        predictions,
        axis=0,
    )[:T]

    if rollout.shape != sim.shape:
        raise RuntimeError(
            f"Rollout shape {rollout.shape} does not "
            f"match ground truth shape {sim.shape}"
        )

    # =====================================================
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
    # =====================================================
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

    print(
        f"Autoregressive relative squared L2 error: "
        f"{relative_squared_l2:.10e}"
    )

    # Frame-wise relative squared L2 error is used only for
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

    predicted_original_indices = (
        represented_original_indices[
            window_frames:
        ]
    )

    # =====================================================
    # Save per-simulation summary
    # =====================================================
    summary_file = os.path.join(
        save_dir,
        "relative_squared_l2.txt",
    )

    with open(
        summary_file,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            f"Simulation: {sim_name}\n"
        )

        file.write(
            f"Test configuration: {config_name}\n"
        )

        file.write(
            f"Model spatial size: {H} x {W}\n"
        )

        file.write(
            f"Spatial start mode: "
            f"{config['spatial_start_mode']}\n"
        )

        file.write(
            f"Temporal skip parameter: "
            f"{config['temporal_skip_param']}\n"
        )

        file.write(
            f"Effective number of frames: {T}\n"
        )

        file.write(
            f"Window size: {window_frames}\n"
        )

        file.write(
            "Channel order: "
            "wave, velocity, pulse_x, pulse_y, "
            "xcoord, ycoord, tcoord\n"
        )

        file.write(
            f"Relative squared L2 error: "
            f"{relative_squared_l2:.10e}\n"
        )

    # =====================================================
    # Save arrays
    # =====================================================
    suffix = (
        "301"
        if config_name == "high_resolution"
        else "121"
    )

    np.save(
        os.path.join(
            save_dir,
            f"{sim_name}_prediction_{suffix}.npy",
        ),
        rollout,
    )

    np.save(
        os.path.join(
            save_dir,
            f"{sim_name}_ground_truth_{suffix}.npy",
        ),
        sim,
    )

    np.save(
        os.path.join(
            save_dir,
            "represented_original_frame_indices.npy",
        ),
        represented_original_indices,
    )

    np.save(
        os.path.join(
            save_dir,
            "frame_relative_squared_l2.npy",
        ),
        frame_relative_squared_l2,
    )

    # =====================================================
    # Frame-wise relative squared L2 error over original
    # frame index
    # =====================================================
    plt.figure(
        figsize=(10, 4)
    )

    plt.plot(
        predicted_original_indices,
        frame_relative_squared_l2,
    )

    plt.xlabel(
        "Original simulation frame index"
    )

    plt.ylabel(
        "Frame-wise relative squared L2 error"
    )

    plt.title(
        f"{sim_name} | 3D U-Net AR relative squared L2 | "
        f"{config_name}"
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

    # =====================================================
    # Plot selected frames
    #
    # Prediction uses the SAME color limits as GT.
    # =====================================================
    original_to_effective = {
        int(original_index): effective_index
        for effective_index, original_index
        in enumerate(
            represented_original_indices
        )
    }

    for original_frame in ORIGINAL_FRAMES_TO_PLOT:

        if original_frame not in original_to_effective:
            continue

        effective_frame = (
            original_to_effective[
                original_frame
            ]
        )

        if effective_frame >= T:
            continue

        ground_truth = sim[
            effective_frame
        ]

        prediction = rollout[
            effective_frame
        ]

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
                frame_squared_error
                / frame_ground_truth_energy_single
            )

        # GT range is used for BOTH panels.
        vmin, vmax = separate_symmetric_limits(
            ground_truth
        )

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(12, 5),
        )

        gt_image = axes[0].imshow(
            ground_truth.T,
            origin="lower",
            cmap="seismic",
            vmin=vmin,
            vmax=vmax,
        )

        axes[0].set_title(
            f"Ground truth\n"
            f"original frame {original_frame}"
        )

        axes[0].axis("off")

        plt.colorbar(
            gt_image,
            ax=axes[0],
        )

        pred_image = axes[1].imshow(
            prediction.T,
            origin="lower",
            cmap="seismic",
            vmin=vmin,
            vmax=vmax,
        )

        axes[1].set_title(
            f"3D U-Net prediction\n"
            f"original frame {original_frame}"
        )

        axes[1].axis("off")

        plt.colorbar(
            pred_image,
            ax=axes[1],
        )

        plt.suptitle(
            f"{sim_name} | {config_name} | "
            f"frame relative squared L2="
            f"{frame_relative_squared_l2_single:.3e}"
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                save_dir,
                f"{sim_name}_original_frame_"
                f"{original_frame:04d}_"
                f"gt_vs_prediction.png",
            ),
            dpi=150,
        )

        plt.close()

    return relative_squared_l2


# =========================================================
# Save one overall relative squared L2 file per resolution
# =========================================================
def save_overall_summary(
    config_name,
    config,
    results,
):
    root_output_directory = config[
        "root_output_directory"
    ]

    overall_results_file = os.path.join(
        root_output_directory,
        config["summary_file"],
    )

    if len(results) > 0:
        overall_mean = float(
            np.mean(
                list(
                    results.values()
                )
            )
        )
    else:
        overall_mean = float("nan")

    with open(
        overall_results_file,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            f"Test configuration: {config_name}\n"
        )

        target = config[
            "target_spatial_size"
        ]

        file.write(
            f"Spatial resolution: "
            f"{target[0]} x {target[1]}\n"
        )

        file.write(
            f"Temporal skip parameter: "
            f"{config['temporal_skip_param']}\n"
        )

        file.write(
            f"Window size: "
            f"{config['window_frames']}\n"
        )

        file.write(
            "Channel order: "
            "wave, velocity, pulse_x, pulse_y, "
            "xcoord, ycoord, tcoord\n"
        )

        file.write(
            "\nPer-simulation relative squared L2 error:\n"
        )

        for simulation_name, rel_error in results.items():
            file.write(
                f"{simulation_name}: "
                f"{rel_error:.10e}\n"
            )

        file.write(
            f"\nMean across simulations: "
            f"{overall_mean:.10e}\n"
        )

    print(
        f"Saved overall summary: "
        f"{overall_results_file}"
    )

    print(
        f"Mean relative squared L2 error across all simulations "
        f"({config_name}): {overall_mean:.6e}"
    )


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":

    script_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    data_folder = os.path.join(
        script_dir,
        "all_test_data",
    )

    model_path = os.path.join(
        script_dir,
        "unet3d_model.pth",
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"U-Net model not found: {model_path}"
        )

    (
        simulation_files,
        xcoord_file,
        ycoord_file,
        tcoord_file,
    ) = get_test_files(
        data_folder
    )

    print("\nSimulation files:")

    for simulation_file in simulation_files:
        print(
            " ",
            os.path.basename(
                simulation_file
            ),
        )

    model = load_model(
        model_path
    )

    # Run the SAME trained U-Net at both resolutions.
    for config_name, config_template in TEST_CONFIGS.items():

        config = dict(
            config_template
        )

        config[
            "root_output_directory"
        ] = os.path.join(
            script_dir,
            config_template[
                "root_output_directory"
            ],
        )

        os.makedirs(
            config[
                "root_output_directory"
            ],
            exist_ok=True,
        )

        print(
            "\n\n################################################"
        )
        print(
            f"STARTING {config_name.upper()} TEST"
        )
        print(
            "################################################"
        )

        results = {}

        for simulation_file in simulation_files:

            simulation_name = os.path.basename(
                simulation_file
            )

            try:
                rel_error = evaluate_autoregressive(
                    sim_file=simulation_file,
                    xcoord_file=xcoord_file,
                    ycoord_file=ycoord_file,
                    tcoord_file=tcoord_file,
                    model=model,
                    config_name=config_name,
                    config=config,
                )

                results[
                    simulation_name
                ] = rel_error

            except Exception as error:
                print(
                    f"\nERROR while testing "
                    f"{simulation_name} "
                    f"at {config_name}: {error}"
                )

                raise

        save_overall_summary(
            config_name=config_name,
            config=config,
            results=results,
        )

    print(
        "\nAll high- and low-resolution "
        "3D U-Net tests completed."
    )

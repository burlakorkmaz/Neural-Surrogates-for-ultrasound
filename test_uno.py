import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import glob
import re
import numpy as np
import torch
import matplotlib.pyplot as plt

from neuralop.models import UNO


# =========================================================
# Device
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# =========================================================
# Shared test settings
# =========================================================
MAX_FRAMES = 300
SPATIAL_START_MODE = "even"


WINDOW_FRAMES = 10
SKIP_PARAM = 3

# These were the original-frame plots.
REQUESTED_ORIGINAL_FRAMES = [
    40, 50, 60, 70, 80, 90, 100, 110, 120, 150, 180, 296
]

COMMON_ORIGINAL_FRAMES_TO_PLOT = [
    frame
    for frame in REQUESTED_ORIGINAL_FRAMES
    if frame % (SKIP_PARAM + 1) == 0
    and frame < MAX_FRAMES
]

print("Requested original plot frames:", REQUESTED_ORIGINAL_FRAMES)
print(
    "Frames represented in BOTH tests:",
    COMMON_ORIGINAL_FRAMES_TO_PLOT,
)


# =========================================================
# Resolution configurations
# =========================================================
TEST_CONFIGS = {
    "high_resolution": {
        "window_frames": WINDOW_FRAMES,
        "temporal_skip_param": SKIP_PARAM,
        "target_spatial_size": (301, 301),
        "output_folder": "test_results_uno3d_high_resolution",
        "summary_filename": "high_resolution_relative_squared_l2.txt",
    },
    "low_resolution": {
        "window_frames": WINDOW_FRAMES,
        "temporal_skip_param": SKIP_PARAM,
        "target_spatial_size": (121, 121),
        "output_folder": "test_results_uno3d_low_resolution",
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
# Spatial resampling helpers
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

    if n_target == n_original:
        return np.arange(n_original, dtype=int)

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
# Test-file discovery
# =========================================================
def get_test_files(data_folder: str):
    data_files = sorted(
        glob.glob(
            os.path.join(
                data_folder,
                "wave_source_*_xyt_kpa.npy",
            )
        )
    )

    if len(data_files) == 0:
        raise FileNotFoundError(
            f"No simulation files found in {data_folder}"
        )

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
    match = re.search(
        r"wave_source_(\d+)_xyt_kpa\.npy",
        basename,
    )

    if match is None:
        raise ValueError(
            f"Could not determine source index from {basename}"
        )

    return match.group(1)


def pulse_files_for_sim(sim_file):
    idx = source_index_from_sim(sim_file)
    folder = os.path.dirname(sim_file)

    pulse_x = os.path.join(
        folder,
        f"pulse_source_{idx}_xcoord.npy",
    )

    pulse_y = os.path.join(
        folder,
        f"pulse_source_{idx}_ycoord.npy",
    )

    if not os.path.exists(pulse_x):
        raise FileNotFoundError(
            f"Missing pulse x file: {pulse_x}"
        )

    if not os.path.exists(pulse_y):
        raise FileNotFoundError(
            f"Missing pulse y file: {pulse_y}"
        )

    return pulse_x, pulse_y


def velocity_file_for_sim(sim_file):
    idx = source_index_from_sim(sim_file)
    folder = os.path.dirname(sim_file)

    source_velocity = os.path.join(
        folder,
        f"velocity_map_source_{idx}.npy",
    )

    global_velocity = os.path.join(
        folder,
        "velocity_map.npy",
    )

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

        return np.tile(
            tcoord_raw[:, :, None],
            (1, 1, T),
        )

    raise ValueError(
        f"Invalid tcoord shape: {tcoord_raw.shape}"
    )


# =========================================================
# U-NO model loading
# =========================================================
def load_model(model_file: str, config_file: str = None):
    """
    Reconstruct the official NeuralOperator U-NO exactly as in training.
    """

    config = {
        "in_channels": 7,
        "out_channels": 1,
        "hidden_channels": 16,
        "lifting_channels": 64,
        "projection_channels": 64,
        "n_layers": 14,
        "uno_out_channels": [
            16, 16,
            24, 24,
            32, 32,
            48, 48,
            32, 32,
            24, 24,
            16, 16,
        ],
        "modes": 48,
        "uno_n_modes": [
            [48, 48, 48]
            for _ in range(14)
        ],
        "uno_scalings": [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 0.5, 0.5],
            [1.0, 1.0, 1.0],
            [1.0, 0.5, 0.5],
            [1.0, 1.0, 1.0],
            [1.0, 0.5, 0.5],
            [1.0, 1.0, 1.0],
            [1.0, 2.0, 2.0],
            [1.0, 1.0, 1.0],
            [1.0, 2.0, 2.0],
            [1.0, 1.0, 1.0],
            [1.0, 2.0, 2.0],
            [1.0, 1.0, 1.0],
        ],
        "horizontal_skips_map": None,
        "window_size": 10,
        "skip_param": 3,
    }

    if config_file is not None and os.path.exists(config_file):
        saved_config = np.load(
            config_file,
            allow_pickle=True,
        ).item()

        for key in config:
            if key in saved_config:
                config[key] = saved_config[key]

        print(
            "Loaded model configuration from:",
            config_file,
        )
        print("Saved configuration:", saved_config)

    else:
        print(
            "Model config file not found; using architecture "
            "settings copied from the supplied U-NO training code."
        )

    in_channels = int(config["in_channels"])
    out_channels = int(config["out_channels"])
    hidden_channels = int(config["hidden_channels"])
    lifting_channels = int(config["lifting_channels"])
    projection_channels = int(config["projection_channels"])
    n_layers = int(config["n_layers"])

    uno_out_channels = [
        int(value)
        for value in config["uno_out_channels"]
    ]

    uno_n_modes = [
        [int(value) for value in layer]
        for layer in config["uno_n_modes"]
    ]

    uno_scalings = [
        [float(value) for value in layer]
        for layer in config["uno_scalings"]
    ]

    horizontal_skips_map = config[
        "horizontal_skips_map"
    ]

    model = UNO(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        lifting_channels=lifting_channels,
        projection_channels=projection_channels,
        positional_embedding=None,
        n_layers=n_layers,
        uno_out_channels=uno_out_channels,
        uno_n_modes=uno_n_modes,
        uno_scalings=uno_scalings,
        horizontal_skips_map=horizontal_skips_map,
        channel_mlp_dropout=0.0,
        channel_mlp_expansion=0.5,
        norm=None,
        fno_skip="linear",
        horizontal_skip="linear",
        channel_mlp_skip="linear",
        factorization=None,
        rank=1.0,
        domain_padding=None,
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

    num_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(f"Loaded U-NO model from: {model_file}")
    print(f"Trainable parameters: {num_parameters:,}")

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
    sim_name = os.path.splitext(
        os.path.basename(sim_file)
    )[0]

    save_dir = os.path.join(
        root_out_dir,
        sim_name,
    )

    os.makedirs(save_dir, exist_ok=True)

    print("\n===================================")
    print(f"Evaluating {sim_name} | {resolution_name}")
    print("===================================")

    # -----------------------------------------------------
    # Load full-resolution wave: (x, y, time)
    # -----------------------------------------------------
    sim_full_xyt = np.load(
        sim_file
    ).astype(np.float32)

    if sim_full_xyt.ndim != 3:
        raise ValueError(
            "Expected wave shape (x,y,t), "
            f"got {sim_full_xyt.shape}"
        )

    original_H, original_W, original_T = sim_full_xyt.shape

    original_T_used = min(
        MAX_FRAMES,
        original_T,
    )

    sim_full_xyt = sim_full_xyt[
        :,
        :,
        :original_T_used,
    ]

    print(
        "Original wave shape used:",
        sim_full_xyt.shape,
    )

    # -----------------------------------------------------
    # Load conditioning maps
    # -----------------------------------------------------
    velocity_file = velocity_file_for_sim(
        sim_file
    )

    pulse_x_file, pulse_y_file = pulse_files_for_sim(
        sim_file
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
                f"{name} shape {arr.shape} does not match "
                f"wave spatial shape {(original_H, original_W)}"
            )

    tcoord_full_xyt = prepare_tcoord_xyt(
        tcoord_raw,
        (
            original_H,
            original_W,
            original_T,
        ),
    )[:, :, :original_T_used]

    # -----------------------------------------------------
    # Spatial resolution handling
    # -----------------------------------------------------
    sim_spatial_xyt = spatial_resample_array(
        sim_full_xyt,
        target_spatial_size,
        SPATIAL_START_MODE,
    ).astype(np.float32)

    velocity_spatial = spatial_resample_array(
        velocity_full,
        target_spatial_size,
        SPATIAL_START_MODE,
    ).astype(np.float32)

    pulse_x_spatial = spatial_resample_array(
        pulse_x_full,
        target_spatial_size,
        SPATIAL_START_MODE,
    ).astype(np.float32)

    pulse_y_spatial = spatial_resample_array(
        pulse_y_full,
        target_spatial_size,
        SPATIAL_START_MODE,
    ).astype(np.float32)

    xcoord_spatial = spatial_resample_array(
        xcoord_full,
        target_spatial_size,
        SPATIAL_START_MODE,
    ).astype(np.float32)

    ycoord_spatial = spatial_resample_array(
        ycoord_full,
        target_spatial_size,
        SPATIAL_START_MODE,
    ).astype(np.float32)

    tcoord_spatial_xyt = spatial_resample_array(
        tcoord_full_xyt,
        target_spatial_size,
        SPATIAL_START_MODE,
    ).astype(np.float32)

    # -----------------------------------------------------
    # Convert (x,y,t) -> (t,x,y), then same temporal skip
    # as training.
    # -----------------------------------------------------
    sim = np.transpose(
        sim_spatial_xyt,
        (2, 0, 1),
    )

    tcoord = np.transpose(
        tcoord_spatial_xyt,
        (2, 0, 1),
    )

    sim = skip_data(
        sim,
        temporal_skip_param,
    ).astype(np.float32)

    tcoord = skip_data(
        tcoord,
        temporal_skip_param,
    ).astype(np.float32)

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
    print(
        "First represented original indices:",
        represented_original_indices[:20],
    )

    if T < 2 * window_frames:
        raise ValueError(
            f"Only {T} effective frames remain, but at least "
            f"{2 * window_frames} are required."
        )

    # -----------------------------------------------------
    # Repeat static maps across effective time
    # -----------------------------------------------------
    velocity = np.tile(
        velocity_spatial[None, :, :],
        (T, 1, 1),
    )

    pulse_x = np.tile(
        pulse_x_spatial[None, :, :],
        (T, 1, 1),
    )

    pulse_y = np.tile(
        pulse_y_spatial[None, :, :],
        (T, 1, 1),
    )

    xcoord = np.tile(
        xcoord_spatial[None, :, :],
        (T, 1, 1),
    )

    ycoord = np.tile(
        ycoord_spatial[None, :, :],
        (T, 1, 1),
    )

    def to_5d_tensor(arr):
        return (
            torch.from_numpy(arr)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )

    velocity_t = to_5d_tensor(velocity)
    pulse_x_t = to_5d_tensor(pulse_x)
    pulse_y_t = to_5d_tensor(pulse_y)
    xcoord_t = to_5d_tensor(xcoord)
    ycoord_t = to_5d_tensor(ycoord)
    tcoord_t = to_5d_tensor(tcoord)

    # -----------------------------------------------------
    # Initial input
    # EXACT training channel order:
    # [wave, velocity, pulse_x, pulse_y, xcoord, ycoord, tcoord]
    # -----------------------------------------------------
    initial_wave = (
        torch.from_numpy(
            sim[:window_frames]
        )
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )

    current_input = torch.cat(
        [
            initial_wave,
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

    if tuple(current_input.shape) != expected_input_shape:
        raise RuntimeError(
            f"Expected input shape {expected_input_shape}, "
            f"got {tuple(current_input.shape)}"
        )

    print(
        "Initial input tensor shape:",
        tuple(current_input.shape),
    )

    predictions = [
        sim[:window_frames].copy()
    ]

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

            expected_output_shape = (
                1,
                1,
                window_frames,
                H,
                W,
            )

            if tuple(output.shape) != expected_output_shape:
                raise RuntimeError(
                    f"Expected output shape {expected_output_shape}, "
                    f"got {tuple(output.shape)}"
                )

            span_effective = min(
                window_frames,
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

                if window.shape[2] < window_frames:
                    missing = (
                        window_frames
                        - window.shape[2]
                    )

                    if window.shape[2] == 0:
                        raise RuntimeError(
                            "Cannot pad an empty conditioning window."
                        )

                    padding = (
                        window[
                            :,
                            :,
                            -1:,
                        ]
                        .repeat(
                            1,
                            1,
                            missing,
                            1,
                            1,
                        )
                    )

                    window = torch.cat(
                        [
                            window,
                            padding,
                        ],
                        dim=2,
                    )

                return window

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
            f"Rollout shape {rollout.shape} does not match "
            f"ground truth {sim.shape}"
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

    predicted_original_indices = (
        represented_original_indices[
            window_frames:
        ]
    )

    print(
        f"Relative squared L2 error ({resolution_name}): "
        f"{relative_squared_l2:.6e}"
    )

    # -----------------------------------------------------
    # Per-simulation summary
    # -----------------------------------------------------
    with open(
        os.path.join(
            save_dir,
            "relative_squared_l2.txt",
        ),
        "w",
        encoding="utf-8",
    ) as file:
        file.write(f"Simulation: {sim_name}\n")
        file.write(f"Resolution test: {resolution_name}\n")
        file.write(
            f"Velocity file: {os.path.basename(velocity_file)}\n"
        )
        file.write(f"Spatial resolution: {H} x {W}\n")
        file.write(f"Spatial start mode: {SPATIAL_START_MODE}\n")
        file.write(
            f"Temporal skip parameter: {temporal_skip_param}\n"
        )
        file.write(f"Window size: {window_frames}\n")
        file.write(f"Effective frames: {T}\n")
        file.write(f"Relative squared L2 error: {relative_squared_l2:.10e}\n")

    # -----------------------------------------------------
    # Save arrays
    # -----------------------------------------------------
    np.save(
        os.path.join(
            save_dir,
            f"{sim_name}_prediction_{H}x{W}.npy",
        ),
        rollout,
    )

    np.save(
        os.path.join(
            save_dir,
            f"{sim_name}_ground_truth_{H}x{W}.npy",
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

    # -----------------------------------------------------
    # Frame-wise relative squared L2 curve vs ORIGINAL
    # simulation index
    # -----------------------------------------------------
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
        f"{sim_name} | U-NO autoregressive relative squared L2 | "
        f"{resolution_name}"
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
    # GT vs prediction plots at SAME original frames
    # for BOTH spatial resolutions.
    # Prediction uses ground-truth colorbar limits.
    # -----------------------------------------------------
    original_to_effective = {
        int(original_index): effective_index
        for effective_index, original_index
        in enumerate(
            represented_original_indices
        )
    }

    for original_frame in (
        COMMON_ORIGINAL_FRAMES_TO_PLOT
    ):
        if original_frame not in original_to_effective:
            raise RuntimeError(
                f"Common original frame {original_frame} "
                f"is not represented in {resolution_name}."
            )

        effective_frame = (
            original_to_effective[
                original_frame
            ]
        )

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

        vmin, vmax = (
            separate_symmetric_limits(
                ground_truth
            )
        )

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(12, 5),
        )

        image_gt = axes[0].imshow(
            ground_truth.T,
            origin="lower",
            cmap="seismic",
            vmin=vmin,
            vmax=vmax,
        )

        axes[0].set_title(
            "Ground truth\n"
            f"original frame {original_frame} "
            f"(effective index {effective_frame})"
        )

        axes[0].axis("off")
        plt.colorbar(
            image_gt,
            ax=axes[0],
        )

        image_prediction = axes[1].imshow(
            prediction.T,
            origin="lower",
            cmap="seismic",
            vmin=vmin,
            vmax=vmax,
        )

        axes[1].set_title(
            "U-NO prediction\n"
            f"original frame {original_frame} "
            f"(effective index {effective_frame})"
        )

        axes[1].axis("off")
        plt.colorbar(
            image_prediction,
            ax=axes[1],
        )

        plt.suptitle(
            f"{sim_name} | {resolution_name} | "
            f"frame relative squared L2="
            f"{frame_relative_squared_l2_single:.3e}"
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                save_dir,
                f"{sim_name}_original_frame_"
                f"{original_frame:04d}_gt_vs_prediction.png",
            ),
            dpi=150,
        )

        plt.close()

    return relative_squared_l2


# =========================================================
# Save one overall relative squared L2 summary per resolution
# =========================================================
def save_resolution_summary(
    results,
    output_dir,
    summary_filename,
    config,
):
    summary_path = os.path.join(
        output_dir,
        summary_filename,
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            f"Resolution: "
            f"{config['target_spatial_size'][0]} x "
            f"{config['target_spatial_size'][1]}\n"
        )

        file.write(
            f"Window frames: "
            f"{config['window_frames']}\n"
        )

        file.write(
            f"Temporal skip parameter: "
            f"{config['temporal_skip_param']}\n"
        )

        file.write(
            "\nPer-simulation relative squared L2 error:\n"
        )

        for name, rel_error in results.items():
            file.write(
                f"{name}: {rel_error:.10e}\n"
            )

        if len(results) > 0:
            overall_mean = float(
                np.mean(
                    list(results.values())
                )
            )

            file.write(
                "\nMean relative squared L2 error across simulations: "
                f"{overall_mean:.10e}\n"
            )

        else:
            overall_mean = np.nan

            file.write(
                "\nMean relative squared L2 error across simulations: NaN\n"
            )

    print(
        f"Saved overall relative squared L2 summary: "
        f"{summary_path}"
    )

    return overall_mean


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

    # Put these next to the test script.
    model_path = os.path.join(
        script_dir,
        "uno3d_model.pth",
    )

    config_path = os.path.join(
        script_dir,
        "uno3d_config.npy",
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"U-NO model not found: {model_path}"
        )

    (
        sim_files,
        xcoord_file,
        ycoord_file,
        tcoord_file,
    ) = get_test_files(
        data_folder
    )

    print("\nSimulation files:")
    for sim_file in sim_files:
        print(
            " ",
            os.path.basename(sim_file),
        )

    model, model_config = load_model(
        model_path,
        config_path,
    )

    trained_window = int(
        model_config.get(
            "window_size",
            WINDOW_FRAMES,
        )
    )

    trained_skip = int(
        model_config.get(
            "skip_param",
            SKIP_PARAM,
        )
    )

    # The low-resolution test reproduces training exactly.
    if trained_window != WINDOW_FRAMES:
        raise ValueError(
            "Low-resolution test window does not match training. "
            f"Training config says {trained_window}, "
            f"low-resolution test uses {WINDOW_FRAMES}."
        )

    if trained_skip != SKIP_PARAM:
        raise ValueError(
            "Low-resolution temporal skip does not match training. "
            f"Training config says {trained_skip}, "
            f"low-resolution test uses {SKIP_PARAM}."
        )

    print("\nIMPORTANT:")
    print("Testing protocol:")
    print(
        f"  low resolution : T={WINDOW_FRAMES}, "
        f"skip_param={SKIP_PARAM}, 121x121"
    )
    print(
        f"  high resolution: T={WINDOW_FRAMES}, "
        f"skip_param={SKIP_PARAM}, 301x301"
    )
    print(
        "The high-resolution test therefore uses 40 consecutive "
        "frames, exactly like the other model tests."
    )
    print(
        "Autoregressive rollout feeds each predicted output block "
        "back as the pressure-history input for the next step."
    )

    all_resolution_means = {}

    for resolution_name, config in TEST_CONFIGS.items():
        print(
            "\n\n"
            "########################################################"
        )
        print(
            f"STARTING {resolution_name.upper()} TEST"
        )
        print(
            "########################################################"
        )

        root_out_dir = os.path.join(
            script_dir,
            config["output_folder"],
        )

        os.makedirs(
            root_out_dir,
            exist_ok=True,
        )

        results = {}

        for sim_file in sim_files:
            sim_key = os.path.basename(
                sim_file
            )

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

        all_resolution_means[
            resolution_name
        ] = overall_mean

        print(
            f"\nCompleted {resolution_name} test."
        )

        print(
            "Per-simulation relative squared L2 errors:"
        )

        for name, rel_error in results.items():
            print(
                f"  {name}: {rel_error:.6e}"
            )

        print(
            "Mean relative squared L2 error across simulations: "
            f"{overall_mean:.6e}"
        )

    print(
        "\n\n"
        "========================================================"
    )
    print(
        "ALL U-NO AUTOREGRESSIVE RESOLUTION TESTS COMPLETED"
    )
    print(
        "========================================================"
    )

    for resolution_name, mean_rel_error in (
        all_resolution_means.items()
    ):
        print(
            f"{resolution_name}: "
            f"mean relative squared L2 error = "
            f"{mean_rel_error:.6e}"
        )

import os
import glob
import re
import numpy as np
import torch
import matplotlib.pyplot as plt
from neuralop.models import FNO3d


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
        "output_folder": "test_results_fno3d_high_resolution",
        "summary_filename": "high_resolution_all_relative_squared_l2.txt",
    },
    "low_resolution": {
        "window_frames": 10,
        "temporal_skip_param": 3,
        "target_spatial_size": (121, 121),
        "output_folder": "test_results_fno3d_low_resolution",
        "summary_filename": "low_resolution_all_relative_squared_l2.txt",
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
# Model loading
# =========================================================
def load_model(model_file: str):
    model = FNO3d(
        n_modes_width=48,
        n_modes_height=48,
        n_modes_depth=48,
        in_channels=7,
        out_channels=1,
        hidden_channels=64,
        n_layers=4,
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
    print(f"Loaded model from: {model_file}")
    return model


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
    # Metrics
    # -----------------------------------------------------
    # Overall relative squared L2 error over the entire predicted signal:
    #
    #     sum((prediction - ground_truth)^2)
    #     ----------------------------------
    #            sum(ground_truth^2)
    #
    # The initial input window is excluded because those frames are
    # ground-truth inputs, not model predictions.
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

    if ground_truth_energy == 0.0:
        relative_squared_l2 = np.nan
    else:
        relative_squared_l2 = float(
            squared_error / ground_truth_energy
        )

    # Frame-wise relative squared L2 values are kept only for the
    # time-index curve and saved frame-wise metric array.
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
        out=np.full_like(frame_squared_errors, np.nan, dtype=np.float64),
        where=frame_ground_truth_energy != 0.0,
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
    # Relative squared L2 curve versus ORIGINAL frame index
    # -----------------------------------------------------
    plt.figure(figsize=(10, 4))
    plt.plot(predicted_original_indices, frame_relative_squared_l2)
    plt.xlabel("Original simulation frame index")
    plt.ylabel("Frame-wise relative squared L2 error")
    plt.title(f"{sim_name} | FNO3D AR relative squared L2 | {resolution_name}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "relative_squared_l2_vs_original_time_index.png"),
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

        frame_squared_error = float(np.sum(
            (prediction - ground_truth) ** 2,
            dtype=np.float64,
        ))
        frame_gt_energy = float(np.sum(
            ground_truth ** 2,
            dtype=np.float64,
        ))

        if frame_gt_energy == 0.0:
            frame_relative_squared_l2_value = np.nan
        else:
            frame_relative_squared_l2_value = (
                frame_squared_error / frame_gt_energy
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
            f"FNO3D prediction\noriginal frame {original_frame} "
            f"(effective index {effective_frame})"
        )
        axes[1].axis("off")
        plt.colorbar(image_prediction, ax=axes[1])

        plt.suptitle(
            f"{sim_name} | {resolution_name} | frame relative squared L2={frame_relative_squared_l2_value:.3e}"
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

        for name, rel_sq_l2 in results.items():
            file.write(f"{name}: {rel_sq_l2:.10e}\n")

        if len(results) > 0:
            overall_mean = float(np.mean(list(results.values())))
            file.write(f"\nMean relative squared L2 across simulations: {overall_mean:.10e}\n")
        else:
            overall_mean = np.nan
            file.write("\nMean relative squared L2 across simulations: NaN\n")

    print(f"Saved overall relative squared L2 summary: {summary_path}")
    return overall_mean


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    data_folder = os.path.join(script_dir, "all_test_data")
    model_path = os.path.join(script_dir, "fno3d_model.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"FNO3D model not found: {model_path}")

    (
        sim_files,
        xcoord_file,
        ycoord_file,
        tcoord_file,
    ) = get_test_files(data_folder)

    print("\nSimulation files:")
    for sim_file in sim_files:
        print(" ", os.path.basename(sim_file))

    # Load the SAME trained model once and use it for both resolutions.
    model = load_model(model_path)

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

            rel_sq_l2 = evaluate_autoregressive(
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

            results[sim_key] = rel_sq_l2

        overall_mean = save_resolution_summary(
            results=results,
            output_dir=root_out_dir,
            summary_filename=config["summary_filename"],
            config=config,
        )

        all_resolution_means[resolution_name] = overall_mean

        print(f"\nCompleted {resolution_name} test.")
        print("Per-simulation relative squared L2 errors:")
        for name, rel_sq_l2 in results.items():
            print(f"  {name}: {rel_sq_l2:.6e}")
        print(f"Mean relative squared L2 across simulations: {overall_mean:.6e}")

    print("\n\n========================================================")
    print("ALL RESOLUTION TESTS COMPLETED")
    print("========================================================")
    for resolution_name, mean_rel_sq_l2 in all_resolution_means.items():
        print(
            f"{resolution_name}: mean relative squared L2 = "
            f"{mean_rel_sq_l2:.6e}"
        )

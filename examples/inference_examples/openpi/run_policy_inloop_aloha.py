"""Real ALOHA rollouts with OpenPI inference and Oopsie browser annotation.

Run on the ROS/Interbotix host; see README_aloha.md for invocation.
"""
from __future__ import annotations

import contextlib
import csv
import dataclasses
import datetime
import importlib
import signal
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import tyro
from openpi_client import image_tools, websocket_client_policy

from oopsie_data_tools.annotation_tool.annotation_schema import outcome_to_success
from oopsie_data_tools.annotation_tool.rollout_annotator import WebRolloutAnnotator
from oopsie_data_tools.utils.robot_profile.robot_profile import load_robot_profile

CAMERA_MAP = {
    "cam_high": "top",
    "cam_left_wrist": "left_wrist",
    "cam_right_wrist": "right_wrist",
}
GRIPPER_INDICES = [6, 13]


@dataclasses.dataclass
class Args:
    robot_profile: Path
    # Checkout on the robot host containing examples/aloha_real.
    openpi_root: Path
    remote_host: str = "127.0.0.1"
    remote_port: int = 8000
    max_timesteps: int = 600
    # 0.5 seconds at 50 Hz; must not exceed the server's chunk length.
    open_loop_horizon: int = 25
    # Smooth selected actions before execution and recording.
    moving_average: bool = False
    # Maximum number of recent action vectors to average.
    moving_average_window: int = 10
    # Exponential decay by action age; zero gives uniform weights.
    moving_average_k: float = 0.1
    # Root directory for saved episodes and evaluation CSV files.
    data_root_dir: Path = Path("./data")
    resume_session_name: Optional[str] = None
    annotator_port: int = 5003
    wait_for_annotation: bool = True
    operator_name: str = "<operator_name>"
    annotator_name: str = "<annotator_name>"


@contextlib.contextmanager
def prevent_keyboard_interrupt():
    """Finish inference or an action/record pair before handling Ctrl+C."""
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt


def _load_real_env(openpi_root):
    root = openpi_root.expanduser().resolve()
    expected = root / "examples" / "aloha_real" / "real_env.py"
    if not expected.is_file():
        raise ValueError(f"--openpi-root must contain examples/aloha_real/real_env.py: {root}")
    sys.path.insert(0, str(root))
    module = importlib.import_module("examples.aloha_real.real_env")
    if Path(module.__file__).resolve() != expected:
        raise RuntimeError(f"ALOHA imported from the wrong checkout: {module.__file__}")
    return module


def _validate_profile(profile):
    if not profile.is_biarm or profile.uses_mobile_base:
        raise ValueError("A stationary bimanual ALOHA profile is required")
    if set(profile.camera_names) != set(CAMERA_MAP.values()):
        raise ValueError(f"Profile camera_names must be {list(CAMERA_MAP.values())}")
    keys = {"joint_position", "gripper_position"}
    if set(profile.robot_state_keys) != keys or set(profile.action_space) != keys:
        raise ValueError("State/action keys must be joint_position and gripper_position")
    for names in (profile.robot_state_joint_names, profile.action_joint_names):
        if names is None or len(names) != 14:
            raise ValueError("Joint names must describe all 14 entries, including grippers")
    if not np.isfinite(profile.control_freq) or profile.control_freq <= 0:
        raise ValueError("control_freq must be positive")


def _extract_observation(obs):
    state = np.asarray(obs["qpos"], dtype=np.float32).copy()
    if state.shape != (14,) or not np.isfinite(state).all():
        raise ValueError(f"Expected finite qpos (14,), got {state.shape}")
    images = {}
    for name in CAMERA_MAP:
        if name not in obs["images"] or obs["images"][name] is None:
            raise ValueError(f"Missing frame: {name}. Check ALOHA ROS camera nodes.")
        image = np.asarray(obs["images"][name])
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError(f"{name}: expected HWC uint8 RGB, got {image.shape}, {image.dtype}")
        # ImageRecorder already requests rgb8. Snapshot before env.step().
        images[name] = image.copy()
    return {"state": state, "images": images}


def _policy_request(obs, instruction):
    return {
        "state": obs["state"].copy(),
        "images": {
            name: np.ascontiguousarray(
                image_tools.resize_with_pad(image, 224, 224).transpose(2, 0, 1)
            )
            for name, image in obs["images"].items()
        },
        "prompt": instruction,
    }


def _validate_chunk(actions, horizon):
    chunk = np.asarray(actions)
    if chunk.ndim != 2 or chunk.shape[1] != 14:
        raise ValueError(f"Expected action chunk (H, 14), got {chunk.shape}")
    if not 1 <= horizon <= len(chunk):
        raise ValueError(f"open_loop_horizon={horizon} must be between 1 and {len(chunk)}")
    if not np.isfinite(chunk).all():
        raise ValueError("Policy returned non-finite actions")
    return chunk


class _ActionMovingAverage:
    """Match aloha_real.main's finite-window exponential action average."""

    def __init__(self, window_size: int, k: float):
        if window_size <= 0:
            raise ValueError("moving_average_window must be positive")
        if not np.isfinite(k) or k < 0:
            raise ValueError("moving_average_k must be finite and non-negative")
        self._history = deque(maxlen=window_size)
        self._k = k

    def reset(self):
        self._history.clear()

    def apply(self, action):
        action = np.asarray(action)
        self._history.append(action.copy())
        actions = np.stack(self._history)
        ages = np.arange(len(actions) - 1, -1, -1, dtype=np.float64)
        weights = np.exp(-self._k * ages)
        weights /= weights.sum()
        return np.sum(actions * weights[:, None], axis=0).astype(action.dtype, copy=False)


def _record_step(annotator, obs, action):
    annotator.record_step(
        observation={
            "image_observation": {CAMERA_MAP[k]: v for k, v in obs["images"].items()},
            "robot_state": {
                "joint_position": obs["state"],
                "gripper_position": obs["state"][GRIPPER_INDICES],
            },
        },
        action={
            "joint_position": action.copy(),
            "gripper_position": action[GRIPPER_INDICES].copy(),
        },
    )


def main(args: Args):
    if args.max_timesteps <= 0 or args.open_loop_horizon <= 0:
        raise ValueError("max_timesteps and open_loop_horizon must be positive")
    smoother = (
        _ActionMovingAverage(args.moving_average_window, args.moving_average_k)
        if args.moving_average else None
    )
    if smoother is not None:
        print(f"Action moving average enabled (window={args.moving_average_window}, "
              f"k={args.moving_average_k:g})")
    profile = load_robot_profile(args.robot_profile)
    _validate_profile(profile)
    real_env = _load_real_env(args.openpi_root)
    policy_client = websocket_client_policy.WebsocketClientPolicy(args.remote_host, args.remote_port)
    metadata = policy_client.get_server_metadata()
    annotator = WebRolloutAnnotator(
        robot_profile=profile,
        data_root_dir=args.data_root_dir,
        port=args.annotator_port,
        wait_for_annotation=args.wait_for_annotation,
        resume_session_name=args.resume_session_name,
        operator_name=args.operator_name,
        annotator_name=args.annotator_name,
    )
    results_dir = args.data_root_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    csv_path = results_dir / f"eval_aloha_{timestamp}.csv"
    period = 1.0 / profile.control_freq
    try:
        annotator.start()
        env = real_env.make_real_env(init_node=True, reset_position=metadata.get("reset_pose"))
        env.reset()
        print(f"Open http://localhost:{args.annotator_port} to start a task.")
        print("Ctrl+C during rollout finishes the episode; Ctrl+C while idle ends the session.")
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=[
                "sample_id", "num_steps", "duration_seconds", "outcome", "success",
            ])
            writer.writeheader()
            file.flush()
            while True:
                instruction = annotator.wait_for_task()
                annotator.reset_episode_recorder()
                if smoother is not None:
                    smoother.reset()
                chunk = None
                chunk_index = 0
                num_steps = 0
                rollout_start = time.monotonic()
                try:
                    for _ in range(args.max_timesteps):
                        step_start = time.monotonic()
                        obs = _extract_observation(env.get_observation())
                        if chunk is None or chunk_index >= args.open_loop_horizon:
                            with prevent_keyboard_interrupt():
                                response = policy_client.infer(_policy_request(obs, instruction))
                            chunk = _validate_chunk(response["actions"], args.open_loop_horizon)
                            chunk_index = 0
                        # Absolute joint positions and continuous grippers: no DROID clipping.
                        action = chunk[chunk_index].copy()
                        with prevent_keyboard_interrupt():
                            if smoother is not None:
                                action = smoother.apply(action)
                            env.step(action.copy())
                            _record_step(annotator, obs, action)
                            num_steps += 1
                        chunk_index += 1
                        remaining = period - (time.monotonic() - step_start)
                        if remaining > 0:
                            time.sleep(remaining)
                except KeyboardInterrupt:
                    print("Rollout interrupted; saving completed steps.")
                finally:
                    # Also save completed steps on camera/inference errors.
                    duration = time.monotonic() - rollout_start
                    if num_steps:
                        annotation = annotator.finish_rollout(instruction=instruction) or {}
                        outcome = annotation.get("outcome", "")
                        writer.writerow({
                            "sample_id": annotator.episode_name,
                            "num_steps": num_steps,
                            "duration_seconds": duration,
                            "outcome": outcome,
                            "success": outcome_to_success(outcome),
                        })
                        file.flush()
                    else:
                        print("No completed steps; skipping empty episode.")
                env.reset()
    except KeyboardInterrupt:
        pass
    finally:
        annotator.stop()
        print(f"Evaluation CSV: {csv_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))

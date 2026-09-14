# Real ALOHA with in-loop annotation

`run_policy_inloop_aloha.py` runs on the robot host with the existing
ROS/Interbotix environment used by `openpi_orl_git/examples/aloha_real`.
Install Oopsie Data Tools, `tyro`, and that checkout's `packages/openpi-client` into
the robot Python environment. Start the ALOHA robot/camera ROS nodes as for
the existing ALOHA client. The policy server runs separately with the config
and checkpoint used to train your ALOHA policy.

From the `oopsie-data-tools` directory on the robot host:

```bash
python examples/inference_examples/openpi/run_policy_inloop_aloha.py \
  --openpi-root /path/to/openpi_orl_git \
  --robot-profile configs/robot_profiles/openpi_aloha_robot_profile.yaml \
  --remote-host 192.168.1.100 --remote-port 8000 \
  --data-root-dir ./data/aloha \
  --operator-name your_name --annotator-name your_name \
  --open-loop-horizon 25 --max-timesteps 600
```

Replace paths, server IP, and names. Verify the profile's robot/gripper identity
and update `policy_name` to identify the deployed policy. It assumes stationary
ViperX 300 dual arms. `--openpi-root` is a checkout on the **robot host**.
The environment creates the ROS node, enables the robot setup, and resets the
arms using server metadata `reset_pose` (or the existing ALOHA default).

Open `http://localhost:5003` on the robot host, submit a task, and annotate each
rollout. Ctrl+C during rollout saves completed steps and proceeds to annotation;
Ctrl+C while waiting for the next task ends the session. Success in the CSV is
derived from the browser annotation. Skipped annotations, or runs with
`--no-wait-for-annotation`, leave outcome/success blank.

The automatically launched annotation server runs in a separate process session
so that a terminal Ctrl+C during rollout does not stop it. Wait for video saving
and the annotation screen before continuing; pressing Ctrl+C again during saving
or annotation can interrupt that work. The runner stops its annotation server
when the session exits. Ctrl+C is a rollout interrupt, not a hardware emergency stop.

The profile's `control_freq` sets the loop target and MP4 FPS (50 by default).
600 steps are nominally 12 seconds; synchronous inference can extend this.
`open_loop_horizon` must be no larger than the returned action chunk length.

The policy receives `state` (14), `images` (RGB uint8 CHW, 224x224), and `prompt`.
Required camera names are `cam_high`, `cam_left_wrist`, `cam_right_wrist`.
Saved camera names are `top`, `left_wrist`, `right_wrist`, respectively.
Images for recording retain their original HWC resolution. No BGR conversion
is applied because this checkout's ALOHA camera subscriber requests `rgb8`.

State and action order is `[left arm 6, left gripper, right arm 6, right gripper]`.
Both 14-vectors are stored under `joint_position`; indices 6 and 13 are also
stored under `gripper_position`. Commands use absolute joint positions and
continuous normalized grippers, matching `real_env.step`. There is no DROID
velocity clipping or gripper binarization.
To enable the same action moving average as `aloha_real/main.py`, add
`--moving-average --moving-average-window 10 --moving-average-k 0.1`.
It averages the most recent selected action vectors, including grippers, with
weights `exp(-k * age)` (newest age is zero). It runs once per control step,
keeps history across inference chunks, and clears history for every episode.
Oopsie records the smoothed command actually sent to the robot. The default is
disabled; `k=0` uses an ordinary window average and window size 1 leaves actions
unchanged. Smoothing can delay responses to action changes.
Snapshots pair pre-action observations with the command passed to `env.step`;
they do not claim that the measured joints have reached the target.

HDF5 and MP4 files are written under `<data-root-dir>/<session>/` at the end
of each rollout. Annotation JSON is written when an annotation is submitted.
CSV summaries are flushed per episode under `<data-root-dir>/results/`.
Completed steps are also finalized on ordinary inference/camera exceptions;
an empty episode is skipped. Frames remain buffered until finalization, so
process termination or power loss before then can lose the active episode.
ORL value predictions and unused action-chunk entries are not included in
the episode; server-side value logging remains available separately.

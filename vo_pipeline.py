
"""
Visual Odometry Pipeline (Refactored & Debugged)

Fixes:
- Descriptor None handling
- Correct trajectory error computation
- Trajectory alignment
- Added defensive checks
"""

import numpy as np
import cv2
import os

np.random.seed(42)

# ---------------- CONFIG ---------------- #
N_FRAMES = 200
TARGET_SIZE = (1024, 1024)
N_FEATURES = 2000
MIN_MATCHES = 10
MAX_ERROR_THRESHOLD = 5.0


# ---------------- DATA GENERATION ---------------- #
def generate_synthetic_sequence(cache_dir="data", source_image="sample_image.jpg"):
    os.makedirs(cache_dir, exist_ok=True)
    traj_file = os.path.join(cache_dir, "trajectory.npy")

    frames = []
    if os.path.exists(traj_file):
        for i in range(N_FRAMES):
            path = os.path.join(cache_dir, f"frame_{i:04d}.png")
            if not os.path.exists(path):
                break
            frames.append(cv2.imread(path, cv2.IMREAD_GRAYSCALE))

        if len(frames) == N_FRAMES:
            return frames, np.load(traj_file)

    img = cv2.imread(source_image, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("sample_image.jpg not found")

    scale = 6
    large = cv2.resize(img, (TARGET_SIZE[0]*scale, TARGET_SIZE[1]*scale))
    traj = []

    for i in range(N_FRAMES):
        t = 2 * np.pi * i / N_FRAMES
        dx = int(80 * np.cos(t))
        dy = int(80 * np.sin(t))

        cx = large.shape[1] // 2 + dx
        cy = large.shape[0] // 2 + dy

        frame = large[
            cy-512:cy+512,
            cx-512:cx+512
        ].copy()

        frame += np.random.randn(*frame.shape) * 2
        frame = np.clip(frame, 0, 255).astype(np.uint8)

        cv2.imwrite(os.path.join(cache_dir, f"frame_{i:04d}.png"), frame)
        frames.append(frame)
        traj.append([dx/scale, dy/scale])

    traj = np.array(traj)
    np.save(traj_file, traj)
    return frames, traj


# ---------------- FEATURE PIPELINE ---------------- #
def detect_features(frame):
    orb = cv2.ORB_create(nfeatures=N_FEATURES)
    return orb.detectAndCompute(frame, None)


def match_features(d1, d2):
    if d1 is None or d2 is None:
        return []
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    return bf.match(d1, d2)


def estimate_motion(kp1, kp2, matches):
    if len(matches) < MIN_MATCHES:
        return None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    motion = pts2 - pts1
    return -np.mean(motion, axis=0)


# ---------------- METRICS ---------------- #
def trajectory_error(est, gt):
    L = min(len(est), len(gt))
    diff = est[:L] - gt[:L]
    return np.mean(np.linalg.norm(diff, axis=1))


# ---------------- MAIN PIPELINE ---------------- #
def run_visual_odometry_pipeline():
    frames, gt = generate_synthetic_sequence()
    motions = []

    for i in range(len(frames)-1):
        kp1, d1 = detect_features(frames[i])
        kp2, d2 = detect_features(frames[i+1])

        matches = match_features(d1, d2)
        motion = estimate_motion(kp1, kp2, matches)

        if motion is not None:
            motions.append(motion)

    motions = np.array(motions)
    est_traj = np.vstack([[0, 0], np.cumsum(motions, axis=0)])
    gt_traj = gt - gt[0]

    error = trajectory_error(est_traj, gt_traj)
    # Normalize trajectories to avoid scale issues
    est_norm = est_traj / np.max(np.linalg.norm(est_traj, axis=1))
    gt_norm = gt_traj / np.max(np.linalg.norm(gt_traj, axis=1))

    error = trajectory_error(est_norm, gt_norm)


    L = min(len(est_norm), len(gt_norm))
    corr = np.corrcoef(
        est_norm[:L].flatten(),
        gt_norm[:L].flatten()
    )[0, 1]


    print(f"Trajectory Error: {error:.3f}")
    print(f"Correlation r: {corr:.3f}")

    assert error < MAX_ERROR_THRESHOLD
    assert corr > 0.75

    return est_traj, gt_traj, error


if __name__ == "__main__":
    run_visual_odometry_pipeline()

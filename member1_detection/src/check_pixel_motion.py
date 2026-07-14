import cv2
import numpy as np

VIDEOS_PATH = "psci_data/Videos"

def compute_motion_activity(video_id, sample_every=5):
    """
    Measures how much the video actually changes frame-to-frame, using
    only raw pixel differences — no detector, no confidence scores, no
    thresholds we chose. This is fully independent evidence.
    """
    cap = cv2.VideoCapture(f"{VIDEOS_PATH}/{video_id}.avi")
    prev_gray = None
    motion_scores = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_every == 0:  # skip some frames for speed
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                motion_scores.append(diff.mean())
            prev_gray = gray
        frame_idx += 1
    cap.release()

    return np.array(motion_scores)


if __name__ == "__main__":
    print(f"{'Video':<8} {'Mean motion':>12} {'Median motion':>14} {'Max motion':>12}")
    print("=" * 50)

    for video_id in [str(i) for i in range(1, 13)]:
        scores = compute_motion_activity(video_id)
        if len(scores) == 0:
            print(f"{video_id:<8} (no frames read)")
            continue
        print(f"{video_id:<8} {scores.mean():>12.2f} {np.median(scores):>14.2f} "
              f"{scores.max():>12.2f}")
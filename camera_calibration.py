#!/usr/bin/env python3
import cv2
import numpy as np
import argparse

def main(checker_rows: int,
         checker_cols: int,
         square_size: float,
         target_frames: int):
    # prepare object points, e.g. (0,0,0), (1,0,0), ... scaled by square_size
    objp = np.zeros((checker_rows*checker_cols,3), dtype=np.float32)
    objp[:,:2] = np.mgrid[0:checker_cols, 0:checker_rows].T.reshape(-1,2)
    objp *= square_size

    objpoints, imgpoints = [], []
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return

    print(f"▶️  Press S to snap when chessboard is detected (need {target_frames} frames)")
    print("▶️  Press C to calibrate once you’ve snapped enough frames")
    print("▶️  Press Q to quit")

    snaps = 0
    pattern = (checker_cols, checker_rows)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(gray, pattern,
                                                   cv2.CALIB_CB_ADAPTIVE_THRESH
                                                   | cv2.CALIB_CB_NORMALIZE_IMAGE)
        display = frame.copy()
        if found:
            cv2.drawChessboardCorners(display, pattern, corners, found)

        cv2.putText(display,
                    f"Snaps: {snaps}/{target_frames}",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,255,0) if found else (0,0,255), 2)
        cv2.imshow("Live Calibration", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q')):
            break
        elif key in (ord('s'), ord('S')) and found:
            # refine and store
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), term)
            objpoints.append(objp)
            imgpoints.append(corners2)
            snaps += 1
            print(f"✅ Captured frame {snaps}")
            if snaps >= target_frames:
                print("✅ Enough frames captured. Press C to calibrate.")
        elif key in (ord('c'), ord('C')) and snaps >= target_frames:
            # perform calibration
            print("🔧 Calibrating...")
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, gray.shape[::-1], None, None
            )
            print(f"RMS error: {ret:.4f}")
            print("Camera matrix:\n", mtx)
            print("Distortion coeffs:", dist.ravel())
            np.savez("calibration.npz", camera_matrix=mtx, dist_coeffs=dist)
            print("✅ Saved to calibration.npz")
            break

    # show undistorted live feed
    try:
        data = np.load("calibration.npz")
        mtx, dist = data["camera_matrix"], data["dist_coeffs"]
    except Exception:
        cap.release()
        cv2.destroyAllWindows()
        return

    print("▶️  Showing undistorted feed. Press Q to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        undistorted = cv2.undistort(frame, mtx, dist, None, new_mtx)
        cv2.imshow("Undistorted Live", undistorted)
        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Live camera calibration with chessboard"
    )
    parser.add_argument('--rows', '-r', type=int, required=True,
                        help='Number of inner corners per column (e.g. 6)')
    parser.add_argument('--cols', '-c', type=int, required=True,
                        help='Number of inner corners per row (e.g. 9)')
    parser.add_argument('--square', '-s', type=float, default=0.025,
                        help='Square size in meters (default: 0.025)')
    parser.add_argument('--frames', '-f', type=int, default=15,
                        help='How many good frames to capture (default: 15)')
    args = parser.parse_args()

    main(args.rows, args.cols, args.square, args.frames)

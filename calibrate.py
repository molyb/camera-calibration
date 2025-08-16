import cv2
import numpy as np
import argparse
import glob
import os
import logging


def save_corners_visualization(
    image, charuco_corners, charuco_ids, original_fname, output_dir
):
    """Draws detected Charuco corners on an image and saves it to a file."""
    if charuco_corners is not None and len(charuco_corners) > 0:
        output_img = image.copy()
        cv2.aruco.drawDetectedCornersCharuco(
            output_img, charuco_corners, charuco_ids, (0, 255, 0)
        )

        base_name = os.path.basename(original_fname)
        name, ext = os.path.splitext(base_name)
        output_fname = os.path.join(output_dir, f"{name}_charuco_corners{ext}")

        cv2.imwrite(output_fname, output_img)
        logging.debug(f"  -> Saved corner visualization to {output_fname}")


def image_coverage_score(charuco_corners, img_size):
    """
    画像内でのCharucoコーナー分布の良さを0..1で返す簡易スコア。
    - バウンディングボックスの面積比
    - 画面中心からの広がり（標準偏差）を組み合わせ
    """
    if charuco_corners is None or len(charuco_corners) < 2:
        return 0.0
    pts = charuco_corners.reshape(-1, 2)
    w, h = img_size
    minxy = pts.min(axis=0)
    maxxy = pts.max(axis=0)
    bbox_area = (maxxy[0] - minxy[0]) * (maxxy[1] - minxy[1])
    img_area = float(w * h)
    area_ratio = max(0.0, min(1.0, bbox_area / (img_area * 0.9)))  # ゆるめに正規化

    center = np.array([w / 2.0, h / 2.0], dtype=np.float32)
    d = np.linalg.norm(pts - center, axis=1)
    spread = np.std(d)
    max_spread = np.hypot(w, h) / 3.0
    spread_ratio = max(0.0, min(1.0, spread / max_spread))

    # 簡易的に平均
    return 0.5 * area_ratio + 0.5 * spread_ratio


def non_redundant_pose_filter(samples, min_translation_pix=40, min_rotation_deg=8.0):
    """
    類似アングル/距離の連続サンプルを間引く簡易フィルタ。
    マーカー重心の移動量と角度（主成分方向）で近接を除去。
    samples: list of dict { 'fname', 'corners'(Nx1x2), 'ids', 'img_size' }
    """
    kept = []

    def pose_signature(s):
        pts = s["corners"].reshape(-1, 2)
        center = pts.mean(axis=0)
        pts_centered = pts - center
        if pts_centered.shape[0] >= 2:
            # 主成分で向き
            cov = np.cov(pts_centered.T)
            eigvals, eigvecs = np.linalg.eig(cov)
            main_vec = eigvecs[:, np.argmax(eigvals)]
            angle = np.degrees(np.arctan2(main_vec[1], main_vec)) % 180.0
        else:
            angle = 0.0
        return center, angle

    signatures = [pose_signature(s) for s in samples]
    for i, s in enumerate(samples):
        c_i, a_i = signatures[i]
        redundant = False
        for j_idx, t_idx in enumerate(kept):
            c_j, a_j = signatures[t_idx]
            if np.linalg.norm(c_i - c_j) < min_translation_pix:
                da = abs(a_i - a_j)
                da = min(da, 180 - da)
                if da < min_rotation_deg:
                    redundant = True
                    break
        if not redundant:
            kept.append(i)
    return [samples[i] for i in kept]


def calibrate_with_pruning(
    all_corners, all_ids, board, img_size, max_iter=2, reproj_sigma=2.5
):
    """
    calibrateCameraCharucoExtended を使ってビューごとの再投影誤差を得て、
    外れ値ビューを除外しつつ再キャリブレーション。
    """
    flags = 0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100, 1e-9)

    # 初回
    ok, cameraMatrix, distCoeffs, rvecs, tvecs, std_int, std_ext, per_view_errors = (
        cv2.aruco.calibrateCameraCharucoExtended(
            charucoCorners=all_corners,
            charucoIds=all_ids,
            board=board,
            imageSize=img_size,
            cameraMatrix=None,
            distCoeffs=None,
            flags=flags,
            criteria=criteria,
        )
    )

    if not ok:
        return ok, None, None, None, None, None

    per_view_errors = np.array(per_view_errors).reshape(-1)
    for it in range(max_iter):
        m = np.median(per_view_errors)
        mad = np.median(np.abs(per_view_errors - m)) + 1e-12
        thresh = m + reproj_sigma * 1.4826 * mad

        keep_idx = [i for i, e in enumerate(per_view_errors) if e <= thresh]
        drop_idx = [i for i, e in enumerate(per_view_errors) if e > thresh]

        if len(drop_idx) == 0 or len(keep_idx) < 3:
            break

        logging.info(
            f"Pruning iteration {it + 1}: dropping {len(drop_idx)} views (error > {thresh:.4f})"
        )

        all_corners = [all_corners[i] for i in keep_idx]
        all_ids = [all_ids[i] for i in keep_idx]

        (
            ok,
            cameraMatrix,
            distCoeffs,
            rvecs,
            tvecs,
            std_int,
            std_ext,
            per_view_errors,
        ) = cv2.aruco.calibrateCameraCharucoExtended(
            charucoCorners=all_corners,
            charucoIds=all_ids,
            board=board,
            imageSize=img_size,
            cameraMatrix=None,
            distCoeffs=None,
            flags=flags,
            criteria=criteria,
        )
        if not ok:
            break
        per_view_errors = np.array(per_view_errors).reshape(-1)

    return ok, cameraMatrix, distCoeffs, rvecs, tvecs, per_view_errors


def process_image(fname, detector, img_size, args, output_dir):
    """
    Processes a single image for calibration.
    Detects board, checks quality, and returns a sample if it's good.
    """
    logging.debug(f"Processing {fname}...")
    img = cv2.imread(fname)
    if img is None:
        logging.warning(f"Warning: Could not read image {fname}. Skipping.")
        return None, img_size

    if img_size is None:
        img_size = img.shape[:2][::-1]  # (width, height)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect the board
    charucoCorners, charucoIds, markerCorners, markerIds = detector.detectBoard(gray)

    # Visualize if requested
    if args.visualize:
        save_corners_visualization(img, charucoCorners, charucoIds, fname, output_dir)

    # Check if the detection is good enough
    if charucoCorners is not None and len(charucoCorners) >= args.min_corners:
        coverage = image_coverage_score(charucoCorners, img_size)
        if coverage >= args.min_coverage:
            sample = {
                "fname": fname,
                "corners": charucoCorners,
                "ids": charucoIds,
                "img_size": img_size,
                "coverage": coverage,
            }
            logging.debug(
                f"  -> Accepted (corners={len(charucoCorners)}, coverage={coverage:.3f})"
            )
            return sample, img_size
        else:
            logging.debug(f"  -> Rejected (low coverage {coverage:.3f})")
            return None, img_size
    else:
        logging.debug("  -> Rejected (insufficient corners)")
        return None, img_size


def main():
    logging.basicConfig(
        level=logging.INFO,  # このレベル以上を出力
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Calibrate camera using ChArUco board and images from a directory (robust to partial views)."
    )
    parser.add_argument(
        "dir", type=str, help="Directory path containing calibration images (JPG)."
    )
    parser.add_argument(
        "--min-corners",
        type=int,
        default=10,
        help="Minimum ChArUco corners per image to accept.",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.12,
        help="Min 0..1 coverage score to accept image.",
    )
    parser.add_argument(
        "--dedup", action="store_true", help="Enable near-duplicate pose filtering."
    )
    parser.add_argument(
        "--save-results",
        type=str,
        default="calibration_results.npz",
        help="Output .npz filename.",
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Save corner visualization images."
    )
    args = parser.parse_args()

    # --- ChArUco Board Setup ---
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    # squaresX, squaresY, squareLength[m], markerLength[m]  -> 実ボードに合わせること
    board = cv2.aruco.CharucoBoard((16, 12), 0.022, 0.016, dictionary)
    detector = cv2.aruco.CharucoDetector(board)

    # --- Output directory for visualizations ---
    output_dir = "detected_corners"
    if args.visualize:
        os.makedirs(output_dir, exist_ok=True)

    # --- Image Loading ---
    image_files = (
        glob.glob(os.path.join(args.dir, "*.jpg"))
        + glob.glob(os.path.join(args.dir, "*.JPG"))
        + glob.glob(os.path.join(args.dir, "*.png"))
        + glob.glob(os.path.join(args.dir, "*.PNG"))
    )
    image_files.sort()
    if not image_files:
        logging.error(f"Error: No image files found in the directory '{args.dir}'.")
        return

    logging.info(f"Found {len(image_files)} images for calibration.")

    # --- Board Detection in all images ---
    allSamples = []
    imgSize = None
    for fname in image_files:
        sample, imgSize = process_image(fname, detector, imgSize, args, output_dir)

        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary)
        corners, ids, rejected, recovered = cv2.aruco.refineDetectedMarkers(
            gray, board, corners, ids, rejected
        )

        if sample:
            allSamples.append(sample)

    if not allSamples:
        logging.error("Error: No acceptable ChArUco detections. Cannot calibrate.")
        return

    # --- Optional: deduplicate similar poses ---
    if args.dedup and len(allSamples) > 8:
        before = len(allSamples)
        allSamples = non_redundant_pose_filter(allSamples)
        after = len(allSamples)
        logging.info(f"Pose de-duplication: {before} -> {after} samples")

    logging.info("Starting calibration...")
    allCorners = [s["corners"] for s in allSamples]
    allIds = [s["ids"] for s in allSamples]
    ok, cameraMatrix, distCoeffs, rvecs, tvecs, per_view_errors = (
        calibrate_with_pruning(
            allCorners, allIds, board, imgSize, max_iter=2, reproj_sigma=2.5
        )
    )

    if not ok or cameraMatrix is None:
        logging.error("\nCalibration failed.")
        return

    logging.info("Calibration successful!")
    logging.info(f"Image size: {imgSize}")
    logging.info(f"Camera Matrix:\n{cameraMatrix}")
    logging.info(f"Distortion Coefficients: {distCoeffs}")

    if per_view_errors is not None:
        logging.debug("Per-view reprojection errors:")
        for i, e in enumerate(per_view_errors):
            logging.debug(f"  {os.path.basename(allSamples[i]['fname'])}: {e:.5f}")

    # Save
    if args.save_results:
        np.savez(
            args.save_results,
            cameraMatrix=cameraMatrix,
            distCoeffs=distCoeffs,
            imageSize=np.array(imgSize, dtype=np.int32),
        )
        logging.info(f"Calibration results saved to '{args.save_results}'")


if __name__ == "__main__":
    main()

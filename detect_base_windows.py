import cv2
import torch
from ultralytics import YOLO
import argparse

def detect_live_webcam(model_path: str='yolo11x.pt', cam_index: int=0):
    """
    Run YOLOv8 on webcam frames, using CUDA if available, and display live detections.
    """
    # On Windows with CUDA: torch.cuda.is_available()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on device: {device}")

    # Load model (no device arg here)
    model = YOLO(model_path)

    # Open the webcam
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)  # CAP_DSHOW often works better on Windows
    if not cap.isOpened():
        print(f"❌ Unable to open camera #{cam_index}")
        return

    print("▶️ Press Q to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to grab frame")
            break

        # Inference—pass device here
        results = model(frame, device=device, conf=0.25, iou=0.45, max_det=100)

        # Annotate and display
        annotated = results[0].plot()
        cv2.imshow('YOLOv11 Live (CUDA)' if device=='cuda' else 'YOLOv11 Live (CPU)', annotated)

        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="YOLOv8 Live Webcam Detection (CUDA/CPU on Windows)"
    )
    parser.add_argument(
        '--model', '-m',
        default='yolo11x.pt',
        help="Path to YOLOv11 model (.pt). Defaults to 'yolo11x.pt'."
    )
    parser.add_argument(
        '--cam', '-c',
        type=int,
        default=0,
        help="Camera index (default 0)."
    )
    args = parser.parse_args()
    detect_live_webcam(args.model, args.cam)

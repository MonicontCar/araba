import argparse, time, os
from pathlib import Path
import cv2
from ultralytics import YOLO
import torch

def parse_args():
    ap = argparse.ArgumentParser(description="YOLOv8-seg realtime inference")
    ap.add_argument("--model", default="araba.pt", help="Model yolun (best.pt)")
    ap.add_argument("--source", default="0", help="0(webcam) | video dosyası | rtsp url")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    ap.add_argument("--iou",  type=float, default=0.50, help="IoU threshold")
    ap.add_argument("--imgsz", type=int, default=640, help="Giriş boyutu")
    ap.add_argument("--save", default="", help="Çıkış video dosyası (örn. out.mp4). Boşsa kaydetmez")
    ap.add_argument("--noshow", action="store_true", help="Pencere açma (headless)")
    ap.add_argument("--stride", type=int, default=1, help="Video frame stride (performans için >1 olabilir)")
    return ap.parse_args()

def main():
    args = parse_args()

    source = 0 if args.source.strip() == "0" else args.source

    use_gpu = torch.cuda.is_available()
    device = 0 if use_gpu else "cpu"
    print(f"Device: {device}  (CUDA={'Yes' if use_gpu else 'No'})")

    model = YOLO(args.model)
    gen = model.predict(
        source=source,
        stream=True,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=device,
        vid_stride=args.stride,
        verbose=False
    )

    writer = None
    save_path = None
    last_t = time.time()
    fps = 0.0

    for res in gen:
        frame = res.plot()
        h, w = frame.shape[:2]

        now = time.time()
        dt = now - last_t
        last_t = now
        inst_fps = 1.0 / max(dt, 1e-6)
        fps = 0.9 * fps + 0.1 * inst_fps if fps > 0 else inst_fps


        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)


        if args.save and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            save_path = args.save
            os.makedirs(Path(save_path).parent if Path(save_path).parent != Path("") else Path("."), exist_ok=True)
            writer = cv2.VideoWriter(save_path, fourcc, 30, (w, h))
            print(f"💾 Kayıt başlatıldı: {save_path}")
        if writer is not None:
            writer.write(frame)


        if not args.noshow:
            cv2.imshow("YOLOv8-seg Realtime", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if writer is not None:
        writer.release()
        print(f"✅ Kayıt tamam: {save_path}")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

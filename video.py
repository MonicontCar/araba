import time
import os
from pathlib import Path

import cv2
from ultralytics import YOLO
import torch

# ================== AYARLAR ==================
VIDEO_PATH = "input.mp4"       # İşlenecek video dosyası
OUTPUT_PATH = "output.mp4"     # Çıkış videosu
MODEL_PATH = "araba.pt"        # YOLO model dosyan (best.pt vs.)
CONF_THRES = 0.25              # Confidence threshold
IOU_THRES = 0.50               # IoU threshold
IMG_SIZE = 640                 # Giriş boyutu
STRIDE = 1                     # Video frame stride
SHOW_WINDOW = True             # Ekranda gösterilsin mi?
# =============================================

def main():
    # Cihaz seçimi (GPU varsa CUDA kullan)
    use_gpu = torch.cuda.is_available()
    device = 0 if use_gpu else "cpu"
    print(f"Device: {device}  (CUDA={'Yes' if use_gpu else 'No'})")

    # Modeli yükle
    model = YOLO(MODEL_PATH)

    # YOLO ile video üzerinde akış tahmini
    gen = model.predict(
        source=VIDEO_PATH,
        stream=True,
        imgsz=IMG_SIZE,
        conf=CONF_THRES,
        iou=IOU_THRES,
        device=device,
        vid_stride=STRIDE,
        verbose=False
    )

    writer = None
    last_t = time.time()
    fps = 0.0

    for res in gen:
        # Çizilmiş frame
        frame = res.plot()
        h, w = frame.shape[:2]

        # FPS hesabı
        now = time.time()
        dt = now - last_t
        last_t = now
        inst_fps = 1.0 / max(dt, 1e-6)
        fps = 0.9 * fps + 0.1 * inst_fps if fps > 0 else inst_fps

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        # VideoWriter sadece bir kere oluşturulsun
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            os.makedirs(
                Path(OUTPUT_PATH).parent
                if Path(OUTPUT_PATH).parent != Path("")
                else Path("."),
                exist_ok=True
            )
            writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, 30, (w, h))
            print(f"💾 Kayıt başlatıldı: {OUTPUT_PATH}")

        writer.write(frame)

        # İsteğe bağlı gösterim
        if SHOW_WINDOW:
            cv2.imshow("YOLOv8-seg Video", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if writer is not None:
        writer.release()
        print(f"✅ Kayıt tamamlandı: {OUTPUT_PATH}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog
import cv2
import os

MODEL_PATH = "araba2.pt"
CONF = 0.25
IOU  = 0.5

def choose_image():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Bir görüntü seç",
        filetypes=[("Görüntüler", "*.jpg *.jpeg *.png *.bmp *.webp"), ("Tümü", "*.*")]
    )
    if not file_path:
        print("⚠️ Dosya seçilmedi, çıkılıyor.")
        exit()
    return file_path

def main():

    image_path = choose_image()
    print(f"🖼️ Seçilen dosya: {image_path}")

 
    model = YOLO(MODEL_PATH)

    results = model.predict(
        source=image_path,
        conf=CONF,
        iou=IOU,
        save=True,    
        imgsz=640
    )
    print("✅ Tahmin tamamlandı.")

    save_dir = results[0].save_dir
    save_name = os.path.basename(results[0].path)
    out_path = os.path.join(str(save_dir), save_name)
    print(f"💾 Kaydedilen çıktı: {out_path}")

if __name__ == "__main__":
    main()

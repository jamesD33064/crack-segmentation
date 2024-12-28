from PIL import Image
from ultralytics import YOLO
import os

# model = YOLO("v8x-seg.pt")
model = YOLO("runs/segment/train2/weights/best.pt")  

dir = "benchmark"
source = [os.path.join(dir, file) for file in os.listdir(dir) if file.lower().endswith((".jpg", ".png"))]

results = model(source)

for i, r in enumerate(results):
    im_bgr = r.plot()  # BGR-order numpy array
    im_rgb = Image.fromarray(im_bgr[..., ::-1])  # 轉換為 RGB-order PIL image
    im_rgb.save(f"predicts/results_{i}.jpg")

    boxes = r.boxes  # YOLO Boxes with coordinates and confidence
    masks = r.masks  # Masks for segmentation (optional)

    with open(f"predicts/results_{i}.txt", "w") as f:
        for box in boxes:
            # 獲取框的座標、信心分數和分類
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # 左上角 (x1, y1) 和右下角 (x2, y2)
            confidence = box.conf.tolist()[0]      # 信心分數
            cls = box.cls.tolist()[0]             # 分類 ID

            # 寫入文字檔，格式：類別 分數 x1 y1 x2 y2
            f.write(f"{cls} {confidence} {x1} {y1} {x2} {y2}\n")
        
        # 如果有 segmentation masks，可選擇性儲存
        if masks is not None:
            for mask_idx, mask in enumerate(masks.data):
                f.write(f"Mask {mask_idx}: {mask.tolist()}\n")
from PIL import Image
from ultralytics import YOLO

# 初始化 YOLO 模型
# model = YOLO("v8x-seg.pt")
model = YOLO("runs/segment/train4/weights/best.pt")  

# 測試影像檔案路徑
# source = "datasets/crack-seg/test/images/1616.rf.c868709931a671796794fdbb95352c5a.jpg"
# source = "DATA_Maguire_20180517_ALL/SDNET2018/D/CD/7001-42.jpg"
source = [
    "benchmark/test1.jpg",
    "benchmark/test2.jpg",
    "benchmark/test3.jpg",
    "benchmark/test4.jpg",
    "benchmark/test5.jpg",
    "benchmark/test6.jpg",
]

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
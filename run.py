from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO("yolo11n-seg.pt")
    # Train the model
    results = model.train(data="datasets/crack-seg/data.yaml", epochs=100, imgsz=640, device=0)
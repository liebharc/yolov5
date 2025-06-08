from ultralytics import YOLO

model = YOLO("yolo11s.pt") # yolov5su yolov8n

model.info()

results = model.train(data="dataset.yml", epochs=100, imgsz=640)
results = model("23_lagenwechsel_2-1.jpeg")

model.export(batch=1,format="onnx",dynamic=True,simplify=True)

print(results)
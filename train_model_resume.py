from ultralytics import YOLO

model = YOLO("/home/ali/yazilim/yolo_model_staj/runs/segment/train-37/weights/last.pt")

results = model.train(
    resume=True,
    epochs=150,
    device=0,
    cache="disk",  # RAM değil disk cache - daha az RAM kullanır
    amp=True,
    plots=True,
    save_period=10,
    val=True,
)
import ultralytics
from ultralytics import YOLO

model = YOLO("/home/ali/yazilim/yolo_model_staj/runs/segment/train-32/weights/best.pt")

results = model.predict(
    source="/home/ali/Masaüstü/test_veri",
    project="/home/ali/yazilim/yolo_model_staj",
    name="test_sonuclari",
    save=True,
    show=False,
    conf=0.40,
    save_txt=True,
    save_conf=True
)

# results, her goruntu icin bir Results nesnesi listesidir
for r in results:
    print(f"Fotoğrafın adı :{r}")
    print(f"Bulunan instance sayisi: {len(r.boxes)}")
    for box, cls_id, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
        class_name = model.names[int(cls_id)]
        print(f"  - {class_name}: guven={conf:.2f}, bbox={box.tolist()}")

    if r.masks is not None:
        print(f"  Maske sayisi: {len(r.masks)}")
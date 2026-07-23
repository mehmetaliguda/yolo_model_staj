from ultralytics import YOLO

model = YOLO("/home/ali/yazilim/yolo_model_staj/runs/segment/train-37/weights/best.pt")

results = model.predict(
    source="/home/ali/Masaüstü/test_veri/test_veri16.jpg",
    save=True,          # sonucu (maske+kutu cizili) runs/segment/predict altina kaydeder
    conf=0.25,          # bu esigin altindaki tahminleri gosterme (varsayilan 0.25, dilerseniz dusurup/artirin)
    imgsz=640,          # egitimdeki imgsz ile ayni olmali
    save_txt=True,      # YOLO formatinda .txt etiket de kaydeder (opsiyonel)
    save_conf=True,     # txt ciktisina guven skorunu da ekler (opsiyonel)
)

# results, her goruntu icin bir Results nesnesi listesidir
for r in results:
    print(f"Bulunan instance sayisi: {len(r.boxes)}")
    for box, cls_id, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
        class_name = model.names[int(cls_id)]
        print(f"  - {class_name}: guven={conf:.2f}, bbox={box.tolist()}")

    if r.masks is not None:
        print(f"  Maske sayisi: {len(r.masks)}")
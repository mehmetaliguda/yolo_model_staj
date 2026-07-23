from ultralytics import YOLO

# En son checkpoint'in gerçek yolunu buraya yaz
last_ckpt = "runs/segment/train/weights/last.pt"

model = YOLO(last_ckpt)
model.train(resume=True, epochs=250)  # epochs sadece toplam hedefi uzatmak için, diğer argümanlar last.pt'den okunur
from ultralytics import YOLO

model = YOLO("yolo26m-seg.pt")

model.train(
    data="/home/ali/Masaüstü/3_etiketli/buzdolabı/etiketli/yolo_devam.yolo26_full_train/dataset_10_kat/dataset.yaml",
    epochs=150,
    patience=25,
    imgsz=1280,               # 1280 -> 800, 6-8GB için gerçekçi üst sınır
    batch=4,                 # sabit, küçük batch
    device=0,
    amp=True,                # mixed precision, VRAM'i ciddi düşürür
    cache=False,
    workers=2,
    lr0=0.001,
    lrf=0.01,
    warmup_epochs=3,
    freeze=15,                # daha fazla katmanı dondur, hem VRAM hem overfit için iyi
    degrees=10.0,
    translate=0.1,
    scale=0.3,
    fliplr=0.5,
    mosaic=0.3,
    copy_paste=0.0,
    hsv_h=0.1,
    hsv_s=0.15,
    hsv_v=0.15,
    close_mosaic=20,
    cls=1.0,
    weight_decay=0.0008,
    dropout=0.0,
    cos_lr=True,
    plots=True,
    save_period=10,
)
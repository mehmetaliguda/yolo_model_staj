from ultralytics import YOLO

# ÖNCEKİ EĞİTİMDEN KALAN EN İYİ MODELİ YÜKLE (Transfer Öğrenmesi / Sıfırdan Başlamama)
model = YOLO("yolo26m-seg.pt")

# MODELİ YENİ VERİLERLE EĞİTMEYE BAŞLA
model.train(
    data="/home/ali/Masaüstü/3_etiketli/buzdolabı/etiketli/yolo_devam.yolo26_full_train/dataset_6_kat/dataset.yaml",
    epochs=150,
    patience=30,
    imgsz=1280,
    rect=True,   # dikdörtgen batch'leme, gereksiz padding'i minimize ede
    batch=0.85,
    device=0,
    lr0=0.0004,
    lrf=0.01,
    warmup_epochs=3,
    freeze=22,
    degrees=10.0,
    translate=0.1,
    scale=0.5,
    shear=0,          # küçük veri setinde gereksiz karmaşıklık
    flipud=0.0,        # çelik şeritte anlamlı değilse kapat
    fliplr=0.5,
    mosaic=0.4,
    mixup=0.0,
    copy_paste=0.1,
    close_mosaic=10,
    cls=1.2,
    weight_decay=0.0005,
    dropout=0.1,
    cos_lr=True,
    plots=True,
    save_period=10,
)
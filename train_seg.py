"""
Mevcut val seti kullanilamaz (tamami defektsiz placeholder) oldugu icin,
gercek etiketli train verisinden rastgele bir dilimi val'e tasiyoruz.
"""

import os
import random
import shutil

TRAIN_IMAGES = "/home/ali/yazilim/yolo_model_staj/veri_setim/train/images"
TRAIN_LABELS = "/home/ali/yazilim/yolo_model_staj/veri_setim/train/labels"
VAL_IMAGES = "/home/ali/yazilim/yolo_model_staj/veri_setim/val/images"
VAL_LABELS = "/home/ali/yazilim/yolo_model_staj/veri_setim/val/labels"

VAL_RATIO = 0.15
SEED = 42

# eski (iste islemez) val icerigini yedekle, ustune yazmayalim
OLD_VAL_BACKUP = "/home/ali/yazilim/yolo_model_staj/veri_setim/val_eski_yedek"


def main():
    random.seed(SEED)

    # eski val'i yedekle
    if os.path.exists(VAL_IMAGES) and os.listdir(VAL_IMAGES):
        os.makedirs(OLD_VAL_BACKUP, exist_ok=True)
        for sub in ("images", "labels"):
            src = f"/home/ali/yazilim/yolo_model_staj/veri_setim/val/{sub}"
            dst = os.path.join(OLD_VAL_BACKUP, sub)
            if os.path.exists(src):
                shutil.move(src, dst)
        print(f"Eski val icerigi yedeklendi: {OLD_VAL_BACKUP}")

    os.makedirs(VAL_IMAGES, exist_ok=True)
    os.makedirs(VAL_LABELS, exist_ok=True)

    label_files = [f for f in os.listdir(TRAIN_LABELS) if f.endswith(".txt")]
    random.shuffle(label_files)

    n_val = int(len(label_files) * VAL_RATIO)
    val_files = label_files[:n_val]

    moved = 0
    for txt_name in val_files:
        base = os.path.splitext(txt_name)[0]

        # goruntu uzantisini bul (jpg/png farkli olabilir)
        img_src = None
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = os.path.join(TRAIN_IMAGES, base + ext)
            if os.path.exists(candidate):
                img_src = candidate
                break
        if img_src is None:
            print(f"UYARI: {base} icin goruntu bulunamadi, atlandi")
            continue

        label_src = os.path.join(TRAIN_LABELS, txt_name)

        shutil.move(img_src, os.path.join(VAL_IMAGES, os.path.basename(img_src)))
        shutil.move(label_src, os.path.join(VAL_LABELS, txt_name))
        moved += 1

    print(f"Toplam {len(label_files)} etiketli goruntuden {moved} tanesi val'e tasindi.")
    print(f"Train'de kalan: {len(label_files) - moved}")


if __name__ == "__main__":
    main()
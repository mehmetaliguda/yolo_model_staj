"""
RLE (EncodedPixels) -> YOLO Segmentation .txt Donusturucu
============================================================
CSV formati: ImageId, ClassId, EncodedPixels (Severstal/Airbus tarzi RLE)
Cikti: her goruntu icin veri_setim/{split}/labels/<isim>.txt

YOLO segmentasyon satir formati:
    class_id x1 y1 x2 y2 x3 y3 ... xn yn   (hepsi 0-1 arasi normalize)

Onemli notlar:
- CSV'deki ClassId 1'den basliyorsa YOLO icin 0'dan baslatmak gerekir (class_id - 1).
- Bir goruntude birden fazla class / birden fazla ayrik bolge (instance) olabilir.
  findContours disjoint (birbirinden kopuk) bolgeleri otomatik ayirir, boylece
  her ayrik bolge YOLO'da ayri bir instance (ayri satir) olarak yazilir.
"""

import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image

# ---------------- AYARLAR ----------------
CSV_PATH = "./sample_submission.csv"                  # ImageId, ClassId, EncodedPixels iceren csv
IMAGES_DIR = "./veri_setim/val/images"  # bu csv'nin karsilik geldigi goruntu klasoru
LABELS_DIR = "./veri_setim/val/labels"  # cikti klasoru (otomatik olusturulur)
MIN_CONTOUR_AREA = 15                     # bu alandan kucuk gurultu bolgeleri atla
APPROX_EPSILON_RATIO = 0.002              # poligon sadelestirme (kucuk = daha detayli kontur)
CLASS_ID_OFFSET = 1                       # CSV'de class'lar 1'den basliyorsa 1, 0'dan basliyorsa 0
IGNORE_CLASS_IDS = {0}                    # gercek nesne olmayan, "defektsiz/arka plan" placeholder ClassId'leri
# ------------------------------------------


def rle_decode(mask_rle: str, shape):
    """
    mask_rle: 'start length start length ...' formatinda string
    shape: (height, width)
    return: (height, width) uint8 binary mask
    """
    s = np.array(mask_rle.split(), dtype=int)
    starts, lengths = s[0::2], s[1::2]
    starts -= 1  # 1-index -> 0-index
    ends = starts + lengths
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    # Kaggle RLE'leri genelde column-major (Fortran) siralamadadir
    return mask.reshape(shape, order="F")


def mask_to_yolo_polygons(mask: np.ndarray):
    """Binary mask -> normalize edilmis poligon listesi (her biri [x1,y1,x2,y2,...])"""
    h, w = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
            continue
        epsilon = APPROX_EPSILON_RATIO * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) < 3:
            continue
        pts = approx.reshape(-1, 2).astype(np.float32)
        pts[:, 0] /= w
        pts[:, 1] /= h
        polygons.append(pts.flatten().tolist())
    return polygons


def main():
    os.makedirs(LABELS_DIR, exist_ok=True)
    df = pd.read_csv(CSV_PATH)

    df_real = df[~df["ClassId"].isin(IGNORE_CLASS_IDS)]
    min_class, max_class = df_real["ClassId"].min(), df_real["ClassId"].max()
    print(f"CSV'deki (placeholder haric) ClassId araligi: {min_class} - {max_class} (offset={CLASS_ID_OFFSET})")
    if min_class - CLASS_ID_OFFSET < 0:
        raise ValueError(
            f"CLASS_ID_OFFSET={CLASS_ID_OFFSET} yanlis: min ClassId={min_class} iken "
            f"sonuc negatif oluyor. CLASS_ID_OFFSET'i {min_class} olarak ayarlayin."
        )

    grouped = df.groupby("ImageId")
    n_images = 0
    n_instances = 0
    skipped = []

    for image_id, group in grouped:
        img_path = os.path.join(IMAGES_DIR, image_id)
        if not os.path.exists(img_path):
            skipped.append(image_id)
            continue

        with Image.open(img_path) as im:
            w, h = im.size  # PIL: (width, height)

        lines = []
        for _, row in group.iterrows():
            raw_class_id = int(row["ClassId"])
            if raw_class_id in IGNORE_CLASS_IDS:
                continue  # defektsiz/arka plan placeholder -> bos etiket kalsin
            class_id = raw_class_id - CLASS_ID_OFFSET
            mask = rle_decode(row["EncodedPixels"], (h, w))
            polygons = mask_to_yolo_polygons(mask)
            for poly in polygons:
                coords_str = " ".join(f"{v:.6f}" for v in poly)
                lines.append(f"{class_id} {coords_str}")
                n_instances += 1

        txt_name = os.path.splitext(image_id)[0] + ".txt"
        with open(os.path.join(LABELS_DIR, txt_name), "w") as f:
            f.write("\n".join(lines))

        n_images += 1

    print(f"Tamamlandi: {n_images} goruntu icin etiket yazildi, toplam {n_instances} instance.")
    if skipped:
        print(f"UYARI: {len(skipped)} goruntu bulunamadi (ornek: {skipped[:5]})")


if __name__ == "__main__":
    main()
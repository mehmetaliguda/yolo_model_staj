from PIL import Image
import os

# Dosya yolları
input_path = "/home/ali/Masaüstü/test_veri1.jpg"
output_path = "/home/ali/Masaüstü/degistirilmis_test_veri1.jpg"

# Görseli aç
try:
    img = Image.open(input_path)
    print(f"Orijinal boyut: {img.size}")
    
    # Yeniden boyutlandır (1600, 256)
    new_size = (1600, 256)
    resized_img = img.resize(new_size, Image.Resampling.LANCZOS)
    
    # Kaydet
    resized_img.save(output_path, quality=95, optimize=True)
    print(f"Yeni boyut: {resized_img.size}")
    print(f"Kaydedildi: {output_path}")
    
except FileNotFoundError:
    print(f"Hata: Dosya bulunamadı - {input_path}")
except Exception as e:
    print(f"Hata oluştu: {e}")
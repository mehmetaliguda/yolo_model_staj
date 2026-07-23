from PIL import Image
import os
from ultralytics import YOLO

# YOLO modelini yükle
model = YOLO("/home/ali/yazilim/yolo_model_staj/runs/segment/train-9/weights/best.pt")

# Giriş ve çıkış yolları
input_path = "/home/ali/yazilim/datasets/crack-seg/images/test/3852.rf.449ecbe5cec6616321519ad9291ee6ea.jpg"
output_folder = "/home/ali/Masaüstü/"
temp_folder = "/home/ali/Masaüstü/temp_chunks/"

# Geçici klasör oluştur
os.makedirs(temp_folder, exist_ok=True)

try:
    # 1. Görseli aç ve yeniden boyutlandır
    img = Image.open(input_path)
    print(f"Orijinal boyut: {img.size}")
    
    # (1600, 256) boyutuna yeniden boyutlandır
    target_size = (1600, 256)
    resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
    resized_img.save(f"{output_folder}resized_full.jpg", quality=95)
    print(f"Yeniden boyutlandırıldı: {target_size}")
    
    # 2. Parçalara ayır
    chunk_width = 1600
    chunk_height = 256
    img_width, img_height = resized_img.size
    
    print(f"\nParçalara ayırma başlıyor...")
    print(f"Görsel boyutu: {img_width}x{img_height}")
    print(f"Parça boyutu: {chunk_width}x{chunk_height}")
    
    chunk_paths = []
    chunk_count = 0
    
    # Yatay ve dikey parçalara ayır
    for y in range(0, img_height, chunk_height):
        for x in range(0, img_width, chunk_width):
            # Parçanın sınırlarını hesapla (son parça daha küçük olabilir)
            right = min(x + chunk_width, img_width)
            bottom = min(y + chunk_height, img_height)
            
            # Parçayı kırp
            chunk = resized_img.crop((x, y, right, bottom))
            chunk_size = chunk.size
            
            # Parçayı geçici klasöre kaydet
            chunk_name = f"chunk_{chunk_count:03d}_{chunk_size[0]}x{chunk_size[1]}.jpg"
            chunk_path = os.path.join(temp_folder, chunk_name)
            chunk.save(chunk_path, quality=95)
            chunk_paths.append(chunk_path)
            
            print(f"Parça {chunk_count:03d}: {chunk_path} - Boyut: {chunk_size}")
            chunk_count += 1
    
    print(f"\n✅ Toplam {chunk_count} parça oluşturuldu.")
    
    # 3. Her parça üzerinde YOLO tahmini yap
    print("\n" + "="*50)
    print("YOLO tahmini başlıyor...")
    print("="*50)
    
    all_results = []
    total_objects = 0
    
    """for i, chunk_path in enumerate(chunk_paths):
        print(f"\n📌 Parça {i:03d} işleniyor: {os.path.basename(chunk_path)}")
        
        # YOLO tahmini yap
        results = model.predict(
            source=chunk_path,
            save=True,          # sonucu runs/segment/predict altına kaydeder
            conf=0.25,          # eşik değeri
            imgsz=640,          # eğitimdeki imgsz ile aynı
            save_txt=True,      # YOLO formatında .txt etiket kaydeder
            save_conf=True,     # txt çıktısına güven skorunu ekler
            project=f"{output_folder}YOLO_results",  # sonuçların kaydedileceği ana klasör
            name=f"chunk_{i:03d}",  # alt klasör adı
            exist_ok=True       # klasör varsa üzerine yaz
        )
        
        # Sonuçları işle
        for r in results:
            obj_count = len(r.boxes) if r.boxes is not None else 0
            total_objects += obj_count
            print(f"  Bulunan instance sayısı: {obj_count}")
            
            # Detayları yazdır
            if r.boxes is not None:
                for box, cls_id, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                    class_name = model.names[int(cls_id)]
                    print(f"    - {class_name}: güven={conf:.2f}, bbox={box.tolist()}")
            
            if r.masks is not None:
                print(f"  Maske sayısı: {len(r.masks)}")
        
        all_results.append(results)
    """
    results = model.predict(
            source=input_path,
            save=True,          # sonucu runs/segment/predict altına kaydeder
            conf=0.25,          # eşik değeri
            imgsz=640,          # eğitimdeki imgsz ile aynı
            save_txt=True,      # YOLO formatında .txt etiket kaydeder
            save_conf=True,     # txt çıktısına güven skorunu ekler
            project=f"{output_folder}YOLO_results",  # sonuçların kaydedileceği ana klasör
            name="result",  # alt klasör adı
            exist_ok=True       # klasör varsa üzerine yaz
    
        )
    for r in results:
        obj_count = len(r.boxes) if r.boxes is not None else 0
        total_objects += obj_count
        print(f"  Bulunan instance sayısı: {obj_count}")
        
        # Detayları yazdır
        if r.boxes is not None:
            for box, cls_id, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                class_name = model.names[int(cls_id)]
                print(f"    - {class_name}: güven={conf:.2f}, bbox={box.tolist()}")
        
        if r.masks is not None:
            print(f"  Maske sayısı: {len(r.masks)}")
    
    all_results.append(results)
    # 4. Özet sonuçları yazdır
    print("\n" + "="*50)
    print("📊 TAHMİN ÖZETİ")
    print("="*50)
    print(f"Toplam parça sayısı: {len(chunk_paths)}")
    print(f"Toplam tespit edilen nesne sayısı: {total_objects}")
    
    # Her parçadaki nesne sayısını göster
    for i, results in enumerate(all_results):
        for r in results:
            obj_count = len(r.boxes) if r.boxes is not None else 0
            print(f"Parça {i:03d}: {obj_count} nesne")
    
    print("\n✅ Tüm işlemler tamamlandı!")
    print(f"✅ Yeniden boyutlandırılmış görsel: {output_folder}resized_full.jpg")
    print(f"✅ Parçalar: {temp_folder}")
    print(f"✅ YOLO sonuçları: {output_folder}YOLO_results/")
    
except FileNotFoundError:
    print(f"❌ Hata: Dosya bulunamadı - {input_path}")
except Exception as e:
    print(f"❌ Hata oluştu: {e}")
    import traceback
    traceback.print_exc()
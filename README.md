YOLO Model Eğitim ve Tahmin Projesi
Bu proje, özel bir iş için YOLO (You Only Look Once) modelinin eğitilmesi ve tahmin yapılması amacıyla oluşturulmuştur.

📁 Dosya Açıklamaları
predict.py
Tek bir görsel üzerinde tahmin yapmak için kullanılır.

Desteklenen formatlar: PNG, JPG, JPEG, BMP, TIFF ve diğer yaygın görsel formatları

Kullanım: Tek bir dosya yolu belirterek çalıştırılır

Çıktı: Tahmin edilen sınıf ve koordinat bilgileri

multiple_predict.py
Belirtilen bir dizindeki tüm görseller üzerinde toplu tahmin yapmak için kullanılır.

Desteklenen formatlar: PNG, JPG, JPEG, BMP, TIFF ve diğer yaygın görsel formatları

Kullanım: Görsellerin bulunduğu dizin yolu belirtilerek çalıştırılır

Çıktı: Tüm görseller için tahmin sonuçları

train_model.py
YOLO modelini sıfırdan eğitmek için kullanılır.

Özellikler:

Veri kümesi hazırlığı

Hiperparametre optimizasyonu

Eğitim metriklerinin takibi

Model ağırlıklarının kaydedilmesi

train_model_resume.py
Kesintiye uğramış model eğitimine devam etmek için kullanılır.

Özellik: YOLO'nun last.pt dosyası sayesinde kaldığı yerden eğitime devam eder

Kullanım: Önceki eğitimden kalan son ağırlıklar yüklenir

Avantaj: Eğitim sürecinin baştan başlatılmasına gerek kalmaz

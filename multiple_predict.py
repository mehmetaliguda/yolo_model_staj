from predict import run_prediction

if __name__ == "__main__":
    # Klasördeki tüm görseller üzerinde çalıştırma örneği
    run_prediction(
        model_path="/home/ali/yazilim/yolo_model_staj/runs/segment/train-32/weights/best.pt",
        source="/home/ali/Masaüstü/test_veri",
        project="/home/ali/yazilim/yolo_model_staj",
        name="test_sonuclari",
        conf=0.40,
    )
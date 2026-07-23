from ultralytics import YOLO


def run_prediction(
    model_path,
    source,
    project=None,
    name=None,
    conf=0.25,
    imgsz=640,
    save=True,
    save_txt=True,
    save_conf=True,
    show=False,
):

    model = YOLO(model_path)

    predict_kwargs = dict(
        source=source,
        save=save,
        show=show,
        conf=conf,
        imgsz=imgsz,
        save_txt=save_txt,
        save_conf=save_conf,
    )
    if project is not None:
        predict_kwargs["project"] = project
    if name is not None:
        predict_kwargs["name"] = name

    results = model.predict(**predict_kwargs)

    # results, her goruntu icin bir Results nesnesi listesidir
    for r in results:
        print(f"Fotoğrafın adı: {r.path}")
        print(f"Bulunan instance sayisi: {len(r.boxes)}")
        for box, cls_id, conf_val in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
            class_name = model.names[int(cls_id)]
            print(f"  - {class_name}: guven={conf_val:.2f}, bbox={box.tolist()}")

        if r.masks is not None:
            print(f"  Maske sayisi: {len(r.masks)}")

    return results


if __name__ == "__main__":
    # Tek görsel üzerinde çalıştırma örneği
    run_prediction(
        model_path="/home/ali/yazilim/yolo_model_staj/runs/segment/train-37/weights/best.pt",
        source="/home/ali/Masaüstü/test_veri/test_veri16.jpg",
        conf=0.25,
        imgsz=640,
    )


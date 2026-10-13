from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model.predict(source="data/test_cube.png",
                        save=True
                        )

for result in results:
    print(result.boxes)
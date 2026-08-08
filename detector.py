from ultralytics import YOLO
import cv2
import os

# Load model safely
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "best.pt")
    return YOLO(model_path)

model = load_model()


def detect_damage(image):
    try:
        # Run inference on CPU
        results = model(image, device="cpu")[0]

        flag = False

        for data in results.boxes.data.tolist():
            confidence = data[4]

            if float(confidence) >= 0.3:
                xmin, ymin, xmax, ymax = map(int, data[:4])

                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(
                    image,
                    "Road Damaged",
                    (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

                flag = True

        if not flag:
            cv2.putText(
                image,
                "Road Repaired",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                3
            )

        return image

    except Exception as e:
        print("ERROR:", e)
        return image

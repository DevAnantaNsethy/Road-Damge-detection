from ultralytics import YOLO
import cv2

model = YOLO("best.pt")

def detect_damage(image):
    results = model(image)[0]

    flag = False  # 👈 to check if damage found

    for data in results.boxes.data.tolist():
        confidence = data[4]

        if float(confidence) >= 0.3:
            xmin, ymin, xmax, ymax = map(int, data[:4])

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(image, "Road Damaged", (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            flag = True  # 👈 damage found

    # ✅ ADD THIS PART
    if not flag:
        cv2.putText(image, "Road Repaired", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

    return image
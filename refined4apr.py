import cv2
import numpy as np

# Open camera
cap = cv2.VideoCapture(0)

# HSV color ranges
color_ranges = {
    "Red1":   (np.array([0, 140, 70], dtype=np.uint8),  np.array([10, 255, 255], dtype=np.uint8)),
    "Red2":   (np.array([170, 140, 70], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8)),
    "Yellow": (np.array([20, 100, 100], dtype=np.uint8), np.array([30, 255, 255], dtype=np.uint8)),
    "Blue":   (np.array([90, 100, 100], dtype=np.uint8), np.array([130, 255, 255], dtype=np.uint8)),
    "Green":  (np.array([35, 100, 100], dtype=np.uint8), np.array([85, 255, 255], dtype=np.uint8))
}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Merge red channels
    masks = {color: cv2.inRange(hsv, lower, upper) for color, (lower, upper) in color_ranges.items()}
    masks["Red"] = cv2.bitwise_or(masks["Red1"], masks["Red2"])
    for k in ["Red1", "Red2"]:
        masks.pop(k)

    # Counters
    triangle_count = 0
    square_count = 0
    rectangle_count = 0
    circle_count = 0

    for color, mask in masks.items():
        # Clean up the mask
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)

        # Find contours
        edges = cv2.Canny(mask, 30, 100)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 300:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            corners = len(approx)

            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            hull_area = cv2.contourArea(cv2.convexHull(cnt))
            solidity = area / hull_area if hull_area > 0 else 0

            shape = None

            if corners == 3 and solidity > 0.80:
                shape = "Triangle"
                triangle_count += 1

            elif corners == 4:
                if 0.90 < aspect_ratio < 1.1:
                    shape = "Square"
                    square_count += 1
                else:
                    shape = "Rectangle"
                    rectangle_count += 1

            elif corners > 4:
                circularity = (4 * np.pi * area) / (peri * peri)
                if circularity > 0.7:
                    shape = "Circle"
                    circle_count += 1

            if shape:
                cv2.drawContours(frame, [approx], -1, (0, 255, 255), 2)
                cv2.putText(frame, f"{color} {shape}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Display counts
    cv2.putText(frame, f"Triangles: {triangle_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Squares: {square_count}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Rectangles: {rectangle_count}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Circles: {circle_count}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Shape Detection", frame)

    # Break on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


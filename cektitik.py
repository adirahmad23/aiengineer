import cv2
from yolov5.detector import YOLOv5Detector
import pathlib

# Mengatur detektor YOLOv5
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

detector = YOLOv5Detector("gpu")
index = "toll_gate.mp4"

def resizeKeepAspectRatio(image, targetWidth):
    height, width = image.shape[:2]
    aspectRatio = width / height
    targetHeight = int(targetWidth / aspectRatio)
    resizedImage = cv2.resize(image, (targetWidth, targetHeight))
    return resizedImage

# Callback untuk menangani klik mouse
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at: x={x}, y={y}")

        # Gambar titik pada posisi yang diklik
        cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)
        cv2.imshow('Frame', frame)

cap = cv2.VideoCapture(index)

# Set callback untuk mouse setelah jendela dibuat
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = resizeKeepAspectRatio(frame, 800)
    cv2.imwrite('temp/{}.png'.format("vid"), frame)
    results = detector.detect('temp/{}.png'.format("vid"))

    for i in results:
        x = i['position']['x']
        y = i['position']['y']
        w = i['position']['w']
        h = i['position']['h']
        x_center = x + w // 2
        y_center = y + h // 2

        # Gambar bounding box dan titik tengah
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, (x_center, y_center), 5, (0, 0, 255), -1)

    # Tampilkan frame dan set callback mouse
    cv2.imshow('Frame', frame)
    cv2.setMouseCallback('Frame', mouse_callback)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

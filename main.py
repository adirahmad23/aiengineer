import cv2
import numpy as np
import time
import os
from yolov5.detector import YOLOv5Detector
import pathlib

# Menyesuaikan Path agar dapat digunakan di Windows
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Inisialisasi YOLOv5Detector menggunakan GPU
detector = YOLOv5Detector("gpu")

# Nama file video yang akan dianalisis
index = "toll_gate.mp4"

# Definisi garis untuk counting sebagai dua titik (start dan end)
line_start = (17, 214)
line_end = (753, 383)
offset = 2  # Jarak toleransi dari garis untuk counting
cooldown_time = 30  # Waktu penundaan dalam detik untuk menghitung objek hanya sekali

# Fungsi untuk meresolusi gambar dengan menjaga rasio aspek
def resizeKeepAspectRatio(image, targetWidth):
    height, width = image.shape[:2]
    aspectRatio = width / height
    targetHeight = int(targetWidth / aspectRatio)
    resizedImage = cv2.resize(image, (targetWidth, targetHeight))
    return resizedImage

# Fungsi untuk memeriksa apakah titik tengah objek melintasi garis yang telah ditentukan
def check_crossing_line(x, y, line_start, line_end, offset):
    line_vector = np.array(line_end) - np.array(line_start)
    point_vector = np.array((x, y)) - np.array(line_start)
    line_length = np.linalg.norm(line_vector)
    line_unit_vector = line_vector / line_length
    projection = np.dot(point_vector, line_unit_vector)
    perpendicular_distance = np.linalg.norm(point_vector - projection * line_unit_vector)
    
    return perpendicular_distance < offset and \
           ((line_start[0] < x < line_end[0]) or (line_end[0] < x < line_start[0]))

# Fungsi untuk menggambar lingkaran dengan transparansi
def draw_transparent_circle(frame, center, radius, color, alpha=0.5):
    overlay = frame.copy()  # Salin frame asli ke overlay
    cv2.circle(overlay, center, radius, color, -1)  # Gambar lingkaran di overlay
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)  # Gabungkan dengan frame asli

# Fungsi utama untuk menjalankan program
def main():
    a = 0
    car_count = 0  # Menghitung jumlah mobil
    bus_count = 0  # Menghitung jumlah bus
    counted_objects = {}  # Menyimpan objek yang sudah dihitung
    object_states = {}  # Menyimpan status objek berdasarkan ID dan waktu
    
    cap = cv2.VideoCapture(index)  # Membuka video

    if not cap.isOpened():
        print("Gagal membuka video")
        return

    while True:
        ret, frame = cap.read()

        if ret:
            frame = resizeKeepAspectRatio(frame, 800)  # Mengubah ukuran frame

            # Simpan frame sebagai gambar sementara
            temp_image_path = 'temp_frame.png'
            cv2.imwrite(temp_image_path, frame)
            
            # Deteksi objek menggunakan YOLOv5
            results = detector.detect(temp_image_path)
            
            # Hapus gambar sementara
            os.remove(temp_image_path)

            current_time = time.time()  # Waktu saat ini untuk pengaturan cooldown
            new_counted_objects = {}
            new_object_states = {}

            for i in results:
                x = i['position']['x']
                y = i['position']['y']
                w = i['position']['w']
                h = i['position']['h']
                
                # Koordinat untuk objek car
                if i["name"] == "car":
                    x_cars = x + w // 2
                    y_cars = y + h // 2
                    object_id = "car_" + str(x_cars) + str(y_cars)
                # Koordinat untuk objek bus
                elif i["name"] == "bus":
                    x_bus = x
                    y_bus = y + h
                    object_id = "bus_" + str(x_bus) + str(y_bus)
                else:
                    continue

                # Cek apakah objek sudah pernah terdeteksi dan telah melintasi garis
                if object_id in counted_objects:
                    if current_time - counted_objects[object_id] < cooldown_time:
                        continue  # Objek belum cooldown, skip deteksi ini
                    else:
                        # Reset status objek jika cooldown sudah berakhir
                        if object_id in object_states:
                            del object_states[object_id]
                        del counted_objects[object_id]

                # Cek jika objek melintasi garis
                if i["name"] == "car" and check_crossing_line(x_cars, y_cars, line_start, line_end, offset):
                    car_count += 1
                    new_counted_objects[object_id] = current_time

                elif i["name"] == "bus" and check_crossing_line(x_bus, y_bus, line_start, line_end, offset) and i["confidence"] > 0.6:
                    bus_count += 1
                    new_counted_objects[object_id] = current_time

                # Gambar bounding box
                if i["name"] == "car":
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(frame, "{} {:.2f}".format(i["name"], i["confidence"]),
                                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    # Menggambar titik yang tidak terlihat
                    # draw_transparent_circle(frame, (x_cars, y_cars), 1, (255, 0, 0), alpha=0)  # Titik tidak terlihat
                
                elif i["name"] == "bus" and i["confidence"] > 0.6:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, "{} {:.2f}".format(i["name"], i["confidence"]),
                                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                
                cv2.putText(frame, "Adi Rahmad R", (610, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            counted_objects.update(new_counted_objects)
            object_states.update(new_object_states)

            # Tampilkan jumlah kendaraan yang terdeteksi
            try:
                cv2.line(frame, line_start, line_end, (0, 255, 0), 2)
                text = f"Cars: {car_count}, Bus: {bus_count}"
                cv2.putText(frame, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            except Exception as e:
                print(f"Error displaying text: {e}")

            cv2.imshow('Camera', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        else:
            # Handle end of video or read error
            a += 1
            print("Gagal membaca frame")
            if a >= 10:
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(index)
                a = 0
                counted_objects.clear()  # Clear counted objects when resetting
                car_count = 0  # Reset car count
                bus_count = 0  # Reset bus count

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

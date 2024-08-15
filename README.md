## Cara Install

1. Clone repository GitHub saya:
   ```bash
   git clone https://github.com/adirahmad23/msibwidyaai.git
2. Clone Yolo V5 jika ada kendala 
      ```bash
   git clone https://github.com/ultralytics/yolov5.git
3. Download bobot yolo nya pada tautan link berikut
   https://drive.google.com/file/d/17oXBidEEiN34SsND4P98USgkut8Iz1-s/view?usp=drive_link
5. Copy dan timpa file program detect.py , detector.py dan yolov5s.pt pada folder yolov5
   
6. install dependensi denga cara buka terminal pada folder yolov5 dan jalankan perintah berikut pada terminal
    ```bash
   pip install -r requirements.txt
7. Install cuda disarankan versi 11.8
8. jika menggunakan cuda versi 11.8 install torch berikut dengan cara, jalankan perintah ini pada terminal
     ```bash
     pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118

## Cara Running Program
1. Buka program main.py
2. jalankan program main.py
    ```bash
   python main.py

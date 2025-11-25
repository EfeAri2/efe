"""
CNN tabanlı yapay zeka algoritması ile yüz tespiti
https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/deploy.prototxt

Yukarıdaki dosyaları indirin
"""

import cv2
import os
import time

# ================================
#   DNN Yüz Algılama Modeli
# ================================
prototxt_path = "deploy.prototxt"
model_path = "res10_300x300_ssd_iter_140000.caffemodel"

# Modeli yükle
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# ================================
#   Kayıt klasörü
# ================================
save_path = "./kisiler"
if not os.path.exists(save_path):
    os.makedirs(save_path)
    print(f"'{save_path}' klasörü oluşturuldu.")

# ================================
#   Kamera açılışı
# ================================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Kamera açılamadı!")
    exit()

print("Program başlatıldı. 'q' ile çıkabilirsiniz.")

last_capture_time = 0
capture_delay = 10

# ================================
#   Döngü
# ================================
while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        print("Kamera okuma hatası, yeniden deneniyor...")
        cap.release()
        time.sleep(0.5)
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        continue

    # Frame boyutları
    (h, w) = frame.shape[:2]

    # =============================================
    #   DNN için blob oluşturma
    #   Blob = giriş görüntüsünün normalize edilmesi
    # =============================================
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),  # model 300x300 bekler
        scalefactor=1.5,
        size=(300, 300),
        mean=(104.0, 177.0, 123.0)  # model ortalaması
    )

    """
    yukarıdaki mean parametresi
    Bu değerler BGR sırasına göre:
    104 → Blue kanalının ortalaması
    177 → Green kanalının ortalaması
    123 → Red kanalının ortalaması
Model eğitilirken veri setindeki tüm yüz resimlerinin kanal ortalamaları hesaplanmış 
ve bu üç değer bulunmuş.
sizin görüntü de ortalama bu değer aralıklarında olursa model iyi çalışıyor
    """

    net.setInput(blob)
    detections = net.forward()  # yüz tahminleri

    # =============================================
    #   Tespit edilen yüzler
    # =============================================
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]  # güven puanı

        # Güven puanı filtresi
        if confidence < 0.6:
            continue

        # Kutu koordinatları (normalize → piksel)
        box = detections[0, 0, i, 3:7] * [w, h, w, h]
        (x1, y1, x2, y2) = box.astype("int")

        # Kutuyu çiz
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{confidence*100:.1f}%", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Foto çekim zaman kontrolü
        current_time = time.time()
        if current_time - last_capture_time > capture_delay:
            filename = f"yuz_{int(current_time)}.jpg"
            full_path = os.path.join(save_path, filename)
            cv2.imwrite(full_path, frame)
            print(f"Yüz kaydedildi: {full_path}")
            last_capture_time = current_time

    cv2.imshow("DNN Yüz Algılama", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
print("Program kapatıldı.")

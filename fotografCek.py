import cv2
import os
import time

# Yüz algılama modeli
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Kayıt klasörü
save_path = "./kisiler"

if not os.path.exists(save_path):
    os.makedirs(save_path)
    print(f"'{save_path}' klasörü oluşturuldu.")

# Kamera başlat
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Kamera açılamadı!")
    exit()

print("Program başladı. 'c' → foto çek (yüz varsa), 'q' → çıkış.")

while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        print("Kamera okuma hatası, yeniden deneniyor...")
        cap.release()
        time.sleep(0.5)
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Algılanan yüzlere dikdörtgen çiz
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("Yuz Algilama", frame)

    key = cv2.waitKey(1) & 0xFF

    # "c" basılırsa → yüz varsa kaydet
    if key == ord('c'):
        if len(faces) > 0:  # yüz bulundu mu?
            file_name = f"yuz_{int(time.time())}.jpg"
            full_path = os.path.join(save_path, file_name)
            cv2.imwrite(full_path, frame)
            print(f"Fotoğraf çekildi: {full_path}")
        else:
            print("Yüz bulunamadı → Fotoğraf çekilmedi.")

    # "q" basılırsa çık
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)

print("Program kapatıldı.")

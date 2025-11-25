import cv2

# 1. Haar Cascade sınıflandırıcısını yükle
# Not: Dosya yolu doğru olmalı! İndirdiğiniz 'haarcascade_frontalface_default.xml' dosyasının adını kullanın.
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')




# Eğer dosya yüklenemezse hata kontrolü
if face_cascade.empty():
    print("Hata: 'haarcascade_frontalface_default.xml' dosyası yüklenemedi. Dosyanın mevcut ve doğru yolda olduğundan emin olun.")
else:
    # 2. Video yakalama nesnesini başlat
    # 0 genellikle yerleşik kamerayı temsil eder. Başka bir kamera kullanıyorsanız bu sayıyı değiştirmelisiniz (örneğin 1, 2, vb.).
    
    try:
    
    
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    except:
        print("Kamera Bulunamadı")
    
    # Kameranın başarıyla açılıp açılmadığını kontrol et
    if not cap.isOpened():
        print("Hata: Kamera açılamadı.")
    else:
        print("Yüz tespiti başlatıldı. Çıkmak için 'q' tuşuna basın.")
        
        # 3. Sonsuz döngüde kareleri oku ve işle
        while True:
            # Kameradan bir kare oku
            ret, frame = cap.read()

            # Kare doğru okunamazsa döngüyü kır
            if not ret:
                print("Kareden veri okunamadı. Çıkılıyor...")
                break

            # Performansı artırmak için kareyi gri tonlamaya dönüştür
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 4. Yüzleri tespit et
            # parameters:
            # - 1.1: scaleFactor (Ne kadar büyükse, o kadar hızlı ama potansiyel olarak daha az doğru)
            # - 4: minNeighbors (Bir yüz adayının korunması için gereken minimum komşu sayısı)
            faces = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30) # Tespit edilecek minimum pencere boyutu
            )

            # 5. Tespit edilen yüzlerin etrafına dikdörtgen çiz
            # faces listesi (x, y, w, h) formatında sonuçlar içerir
            for (x, y, w, h) in faces:
                # Dikdörtgen çiz: (resim, başlangıç noktası, bitiş noktası, renk (BGR), kalınlık)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # İstenirse yüzün üstüne "Yüz" yazısı eklenebilir
                cv2.putText(frame, "Reis", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


            # 6. Sonuç karesini göster
            cv2.imshow('Yuz Tespiti', frame)

            # 'q' tuşuna basılırsa döngüden çık
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 7. Temizlik
        cap.release()
        cv2.destroyAllWindows()
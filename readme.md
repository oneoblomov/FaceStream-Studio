# **📘 Kullanıcı El Kitapçığı: Yüz Analizi ve Konuşma Takip Uygulaması**

## **1. Giriş ve Amaç**

Bu uygulama, canlı kamera akışı veya yüklenen video dosyaları üzerinden:

- **Konuşan kişinin kimliğini** tanımlama,
- **Konuşma sürelerini** hesaplama,
- **Duygu durumu** (mutlu, rahatsız, nötr) tahmini yapma  
  yeteneğine sahiptir. Proje, görüntü işleme ve makine öğrenimi tekniklerini kullanarak gerçek zamanlı analiz sunar.

## **2. Sistem Gereksinimleri**

- **İşletim Sistemi:** Windows 10/11, macOS 12+, Linux (Ubuntu 20.04+ önerilir)
- **Python:** 3.8 veya üzeri
- **Donanım:**
  - GPU (CUDA desteği önerilir, ancak CPU ile de çalışır)
  - Web kamerası (canlı analiz için)
- **Kütüphaneler:**
  - `opencv-python`, `streamlit`, `ultralytics`, `face-recognition`, `mediapipe`, `numpy`

## **3. Kurulum Talimatları**

### **Adım 1: Kodun İndirilmesi**

- Proje dosyalarını GitHub üzerinden indirin veya `.zip` olarak kaydedin.

### **Adım 2: Sanal Ortam Oluşturma (Opsiyonel)**

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

### **Adım 3: Bağımlılıkların Yüklenmesi**

```bash
pip install opencv-python streamlit ultralytics face-recognition mediapipe numpy
```

### **Adım 4: Yüz Veri Kümesinin Hazırlanması**

- `faces` klasörüne tanınmasını istediğiniz kişilerin fotoğraflarını ekleyin (`.jpg`, `.png` formatında).  
  Örnek:

  ```plaintext
  ├── faces/
  │   ├── ahmet.jpg
  │   └── mehmet.png
  ```

#### **Adım 5: Uygulamayı Başlatma**

```bash
streamlit run app.py
```

## **4. Kullanım Kılavuzu**

### **Arayüze Erişim**

Uygulama başlatıldığında tarayıcınızda otomatik olarak bir sekme açılır.

### **Mod Seçimi**

- **Canlı Kamera Modu:**

  1. Sol menüden **"Camera"** seçin.
  2. **"Start"** düğmesine basarak analizi başlatın.
  3. **"Stop"** düğmesi ile durdurun.

  - **Çıktı:** Ekranda gerçek zamanlı yüz tanıma, duygu durumu ve konuşma süreleri görüntülenir.

- **Video Dosyası Modu:**
  1. Sol menüden **"Video"** seçin.
  2. **"Choose a video"** ile bir video yükleyin (mp4, avi, mov).
  3. Analiz otomatik başlar.
  - **Çıktı:** Analiz tamamlandığında konsolda konuşma süreleri listelenir.

## **5. Sonuçların Yorumlanması**

- **Ekran Çıktıları:**
  - **Yeşil Kutu:** Mutlu ifade.
  - **Kırmızı Kutu:** Rahatsız ifade.
  - **Gri Kutu:** Nötr ifade.
  - **Alt Bilgi:** Kişi adı ve toplam konuşma süresi.
- **Konsol Çıktıları:**

  ```plaintext
  --- Konuşma Süreleri ---
  Ahmet: 34.25 saniye
  Mehmet: 22.50 saniye
  ```

## **6. Sorun Giderme**

- **Kamera Açılmıyorsa:**

  - Güvenlik ayarlarınızda kamera erişimine izin verin.
  - `cv2.VideoCapture(0)` satırındaki "0" değerini farklı bir indeksle değiştirin (örn. 1).

- **Bağımlılık Hataları:**

  ```bash
  pip install --upgrade numpy  # Belirli kütüphaneleri güncelleyin
  ```

- **Yüz Tanıma Çalışmıyorsa:**
  - `faces` klasöründeki görsellerin yüksek çözünürlüklü ve tek kişi içerdiğinden emin olun.

## **7. Bilinen Sınırlamalar**

- Duygu tahmini, aydınlatma ve kamera açısından etkilenebilir.
- Yüksek çözünürlüklü videolarda performans düşebilir.

## **8. İletişim ve Destek**

- **Hata Bildirimi:** [GitHub Issues](https://github.com/oneoblomov/yourproject/issues)
- **E-posta:** <muhakaplan@hotmail.com>

## **9. Ekler**

### **Parametre Açıklamaları**

- **`SPEAKING_THRESHOLD`:** Konuşma tespiti için dudak hareketi eşiği (varsayılan: `0.05`).
- **`FACE_RECOGNITION_THRESHOLD`:** Yüz tanıma hassasiyeti (varsayılan: `0.6`).

### **Geliştirici Notları**

- Yeni duygu sınıfları eklemek için `analyze_emotion()` metodunu değiştirin.
- Model dosyalarını `yolov11l-face.pt` yerine özel eğitilmiş modellerle değiştirebilirsiniz.

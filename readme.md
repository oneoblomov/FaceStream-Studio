# 📘 Yüz Analiz Uygulaması Kullanıcı Kılavuzu

## Face Analysis Application User Guide

Bu kılavuz, **Yüz Analiz Uygulaması** (FaceStream Studio) ile ilgili kurulum, kullanım ve özellikler hakkında detaylı bilgi sunar.

---

## 📝 Uygulama Özeti

**FaceStream Studio**, gerçek zamanlı yüz tanıma, duygu analizi ve konuşma süresi takibi sağlayan bir Streamlit tabanlı uygulamadır. Kamera veya video dosyası üzerinden analiz yapabilir, yeni yüzler ekleyebilir ve sonuçları dışa aktarabilirsiniz.

---

## 📥 Kurulum

### Gereksinimler

- Python 3.8 veya üzeri
- `requirements.txt` dosyasında listelenen kütüphaneler

### Kurulum Adımları

1. **Sanal Ortam Oluşturun (Önerilir):**  
   Proje bağımlılıklarını yönetmek için sanal ortam kullanmanız önerilir.

   ```bash
   python -m venv myenv
   source myenv/bin/activate  # Linux/Mac
   myenv\Scripts\activate    # Windows
   ```

2. **Bağımlılıkları Yükleyin:**  
   Gerekli kütüphaneleri yüklemek için aşağıdaki komutu çalıştırın:

   ```bash
   pip install -r requirements.txt
   ```

3. **Model Dosyalarını Kontrol Edin:**  
   Uygulamanın çalışabilmesi için `src` klasöründe YOLO model dosyaları (`.pt` uzantılı) bulunmalıdır.  
   Örnek klasör yapısı:

   ```
   proje_dizini/
   ├── src/
   │   ├── yolov8n-face.pt
   │   └── ...
   ├── Analyzer.py
   ├── app.py
   └── requirements.txt
   ```

---

## 🚀 Uygulamanın Başlatılması

Terminalde aşağıdaki komutu çalıştırarak uygulamayı başlatın:

```bash
streamlit run app.py
```

---

## 🖥️ Ana Arayüz ve Temel Kullanım

- **Çalışma Modu Seçimi:**  
  Sol menüden **📷 Kamera** veya **🎞️ Video** modunu seçebilirsiniz.
- **Ayarlar:**  
  `⚙️ Ayarlar` menüsünden model parametrelerini ve görüntüleme seçeneklerini özelleştirebilirsiniz.

---

## 📷 Kamera Modu

### Özellikler

- **Canlı Analiz:** Kamera ile gerçek zamanlı yüz tanıma, duygu analizi ve konuşma süresi takibi.
- **Yüz Ekleme:** Sağ panelden yeni yüzler ekleyebilirsiniz.

### Kullanım Adımları

1. **"Başlat"** düğmesine tıklayın.
2. Kamera görüntüsü ekranda belirecektir.
3. **"Durdur"** düğmesi ile analizi sonlandırabilirsiniz.

---

## 🎞️ Video Modu

### Özellikler

- **Video Yükleme:** MP4, AVI veya MOV formatında video dosyalarını analiz edebilirsiniz.
- **Sonuçlar:** Konuşma süreleri ve yüz tanıma sonuçları sağ panelde görüntülenir.

### Kullanım Adımları

1. **"Video yükle"** butonuyla bir dosya seçin.
2. **"Analiz Başlat"** düğmesine tıklayın.
3. Analiz tamamlandığında sonuçlar sağ panelde gösterilecektir.

---

## 👤 Yüz Yönetimi

### Yeni Yüz Ekleme

1. Sağ paneldeki **"Yeni Yüz Ekle"** bölümüne gidin.
2. Bir fotoğraf yükleyin ve isim girin.
3. **"Yüz Ekle"** düğmesine tıklayın.

### Kayıtlı Yüzleri Silme

- Yüzlerin yanındaki **❌** simgesine tıklayarak silebilirsiniz.

---

## ⚙️ Ayarlar

### Model ve Algılama Parametreleri

- **Yüz Eşleşme Eşiği:** Yüz tanıma hassasiyetini ayarlar (düşük değer = daha hassas).
- **Maksimum Yüz Sayısı:** Aynı anda tespit edilecek maksimum yüz sayısı.
- **Frame Atlatma:** İşlem hızını artırmak için analiz edilmeyen kare sayısı.
- **Konuşma Eşiği:** Konuşma tespiti için dudak mesafesi eşiği.

### Görüntüleme Seçenekleri

- **İsimleri Göster:** Tanınan yüzlerin isimlerini gösterir.
- **Konuşma Süresini Göster:** Her yüz için konuşma süresini gösterir.
- **Duygu Analizi:** Yüz ifadelerini gösterir.
- **Kutu Göster:** Yüzlerin etrafında kutu çizer.

### Dil ve Tema

- **Dil Seçimi:** Uygulama arayüz dilini değiştirin.

---

## 🛠️ Sorun Giderme

### Sık Karşılaşılan Sorunlar ve Çözümleri

1. **Model Dosyaları Bulunamadı:**

   - `src` klasörünün doğru konumda olduğundan emin olun.
   - Model dosyalarını [resmi YOLO reposundan](https://github.com/ultralytics/ultralytics) indirip ilgili klasöre ekleyin.

2. **Kamera Açılmıyor:**

   - Başka bir uygulamanın kamerayı kullanmadığından emin olun.
   - Terminalde kamera erişim izinlerini kontrol edin.

3. **Bağımlılık Hataları:**

   - Sanal ortam kullanıyorsanız yeniden etkinleştirin.
   - Tüm kütüphanelerin doğru sürümlerini yükleyin:

     ```bash
     pip install --upgrade -r requirements.txt
     ```

---

## 📊 Çıktılar ve Anlamları

- **Duygu Analizi:** Yüz ifadeleri `HAPPY`, `ANNOYED`, `SURPRISED` veya `NEUTRAL` olarak sınıflandırılır.
- **Konuşma Süresi:** Her yüz için toplam konuşma süresi (saniye cinsinden) gösterilir.
- **Yüz Kutuları:** Tanınan yüzler yeşil kutularla işaretlenir ve isimleri görüntülenir.

---

## 📤 Sonuçları Dışa Aktarma

- Sağ menüden analiz sonuçlarını CSV dosyası olarak indirebilirsiniz.

---

## 📞 İletişim ve Destek

Her türlü soru, öneri veya teknik destek talepleriniz için lütfen aşağıdaki e-posta adresiyle iletişime geçin:

**E-posta:** [muhakaplan@hotmail.com](mailto:muhakaplan@hotmail.com)

---

Bu kılavuz, uygulamanın temel işlevlerini etkin şekilde kullanmanıza yardımcı olacaktır.

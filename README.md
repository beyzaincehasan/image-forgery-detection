# 🔍 Görüntü Sahteciliği Tespit Sistemi (Image Forgery Detection)

Bu proje; dijital görüntüler üzerinde gerçekleştirilen ekleme (splicing) ve manipülasyon işlemlerini tespit etmek amacıyla geliştirilmiş, derin öğrenme ve bilgisayarlı görü (computer vision) tekniklerini bir arada kullanan **hibrit adli analiz aracıdır**. Uygulama, modern ve kullanıcı dostu bir web arayüzü sunan **Streamlit** kütüphanesi ile geliştirilmiştir.

---

## 🛠️ Teknolojik Omurga ve Çalışma Mantığı

Sistem, yüklenen görüntüler üzerinde iki aşamalı bir doğrulama altyapısı çalıştırır:

1. **Hata Seviyesi Analizi (Error Level Analysis - ELA):** Görüntü ön işleme aşamasında $Q = 95$ kalite katsayısında yeniden sıkıştırılır. Orijinal pikseller ile sıkıştırma sonrası oluşan kayıplar arasındaki matris farkı hesaplanır. Sonradan manipüle edilmiş alanlar ELA çıktısında yüksek parlaklık değerleriyle kendini belli eder.
2. **Ensemble (Topluluk) Derin Öğrenme:** ELA filtresinden geçen öznitelikler, eş zamanlı olarak **Xception** ve **MobileNetV2** evrişimsel sinir ağı (CNN) modellerine beslenir. Sistem, iki modelin ürettiği tahmin skorlarının aritmetik ortalamasını alarak nihai kararı üretir:
   $$\text{Nihai Güven Skoru} = \frac{\text{Xception Skoru} + \text{MobileNetV2 Skoru}}{2}$$
3. **Dokusal Öznitelik Çıkarımı:** Derin öğrenme modellerini desteklemek adına bilgisayarlı görü algoritmaları olan **SIFT**, **AKAZE** ve **ORB** ile görüntü yüzeyi taranarak yapısal anahtar nokta (keypoint) deformasyonları haritalandırılır.

---

## 📊 Proje Yönetimi ve Emek Kestirimi (COCOMO I)

Yazılım proje yönetimi standartlarına uygun olarak, projenin geliştirme süreci **Basic COCOMO (Yarı Ayrık / Semi-Detached)** modeli kullanılarak analitik olarak doğrulanmıştır:

* **Proje Boyutu:** 1.0 KLOC (1000 Satır Kaynak Kod)
* **Hesaplanan İş Gücü (Effort):** $3.0 \times (1.0)^{1.12} = 3.0 \text{ Adam-Ay}$
* **Teorik Geliştirme Süresi:** $2.5 \times (3.0)^{0.35} = 3.67 \text{ Ay}$
* **Nihai Toplam Emek:** $3.0 \times 160 \text{ Saat} = \mathbf{480\text{ Adam-Saat}}$

---

## 🚀 Kurulum ve Çalıştırma

Projenin yerel sunucunuzda kararlı çalışabilmesi için Python 3.10.x sürümü önerilir.

```bash
# 1. Depoyu klonlayın
git clone [https://github.com/kullanici_adi/repo_adi.git](https://github.com/kullanici_adi/repo_adi.git)
cd repo_adi

# 2. İzole bir Python sanal ortamı oluşturun ve aktifleştirin
python -m venv forgery_env
# Windows için:
forgery_env\Scripts\activate
# Linux/macOS için:
source forgery_env/bin/activate

# 3. Gerekli bağımlılık kütüphanelerini yükleyin
pip install -r requirements.txt

# 4. Uygulamayı başlatın
streamlit run app.py

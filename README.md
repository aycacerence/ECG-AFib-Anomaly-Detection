# Atriyal Fibrilasyon Tespit Sistemi (1D-CAE)

## Proje Özeti
Bu proje, yüksek frekanslı (400 Hz) Elektrokardiyogram (EKG) sinyallerinde Atriyal Fibrilasyon (AFib) anomalilerini otonom olarak tespit etmek amacıyla geliştirilmiş Bir Boyutlu Evrişimli Otokodlayıcı (1D Convolutional Autoencoder - 1D-CAE) tabanlı bir makine öğrenmesi sistemidir. Proje, donanımsal kaynakları sınırlı giyilebilir sağlık teknolojileri ve gerçek zamanlı hastane izleme (monitoring) sistemleri için yüksek hassasiyetli, düşük gecikmeli bir çözüm sunmayı hedeflemektedir.

## Temel Özellikler
* **Yüksek Performans:** Kapsamlı hiperparametre optimizasyonu (Grid Search) ile elde edilen **0.9310 ROC-AUC** skoru.
* **Gelişmiş Veri Ön İşleme (Pseudo-Labeling):** İstatistiksel varyans ve genlik eşikleri kullanılarak 19.000 adet etiketsiz ham EKG verisi üzerinden yarı-gözetimli (semi-supervised) pseudo-labeling süreci yürütülmüş, saf 'Normal' veri seti otonom olarak genişletilmiştir.
* **Hafif Mimarî (Lightweight Architecture):** Giyilebilir cihaz (wearable IoT) entegrasyonuna uygun olacak şekilde, parametre maliyeti düşük ve çıkarım (inference) hızı maksimize edilmiş 1D-CAE dizaynı.
* **Klinik Simülasyon Arayüzü:** Gerçek zamanlı EKG analizi, Z-Skoru normalizasyonu ve dinamik güven skoru (% Confidence) hesaplamaları sunan, Streamlit tabanlı profesyonel klinik gözlem paneli.
* **SQLite Veritabanı Entegrasyonu:** Gerçekleştirilen tüm analizlerin protokol numarası, MSE skoru ve teşhis kararı ile birlikte tarih damgalı olarak kaydedildiği, geçmişe dönük sorgulanabilir klinik kayıt sistemi.

## Teknik Mimari ve Anomali Tespiti İşleyişi
Sistem, gözetimsiz anomali tespiti (Unsupervised Anomaly Detection) prensibine dayanmaktadır. 1D-CAE modeli, yalnızca sağlıklı (Normal Sinüs Ritmi) EKG sinyalleri ile eğitilmiştir. Modelin kodlayıcı (Encoder) katmanları sağlıklı sinyallerin morfolojik özelliklerini düşük boyutlu bir gizli uzaya (Latent Space) sıkıştırırken; kod çözücü (Decoder) katmanları bu özellikleri kullanarak orijinal sinyali yeniden inşa eder.

Test aşamasında (Inference), sisteme daha önce görülmemiş bir Atriyal Fibrilasyon (AFib) sinyali beslendiğinde, model yalnızca sağlıklı yapıları öğrenmiş olduğu için patolojik sinyali doğru bir şekilde yeniden inşa edemez. Bu durum, giriş sinyali ile yeniden inşa edilen sinyal arasındaki Ortalama Karesel Hatanın (Mean Squared Error - MSE) dramatik bir biçimde artmasına neden olur. Sistem, önceden belirlenmiş bir optimal karar eşiği (Threshold = 0.092) üzerinden bu hatayı değerlendirerek sınıflandırma işlemini gerçekleştirir.

## Klasör Yapısı
```text
Proje/
│
├── app.py                      # Streamlit tabanlı dinamik klinik izleme arayüzü
├── train_final_model.py        # 1D-CAE modelinin optimal parametrelerle nihai eğitim betiği
├── model_comparison.py         # 1D-CAE ve ConvLSTM mimarilerinin kıyaslama testi betiği
├── pseudo_label_pipeline.py    # Etiketsiz verilerin filtrelenmesi ve pseudo-label üretimi
├── grid_search.py              # Hiperparametre optimizasyon sürecini yürüten betik
├── fast_plots.py               # IEEE akademik makale standartlarında (300 DPI) görselleştirme aracı
├── requirements.txt            # Proje bağımlılıkları ve kütüphane versiyonları
│
├── Ara Rapor/                  # Ara Sınav (V1) dönemine ait arşivlenmiş betikler ve PDF raporlar
│   ├── ecg_afib_autoencoder.py # Projenin ilk fazına ait monolitik V1 eğitim betiği
│   ├── lstm_autoencoder.py     # Alternatif V1 LSTM-Autoencoder modeli
│   └── sanity_check.py         # Çevresel ortam (kütüphane/donanım) doğrulayıcı
│
├── Dataset/                    # Ham EKG sinyallerini barındıran veri klasörü
│   └── traindata.csv           # Test için kullanılan etiketli sinyaller (AFib ve Normal)
│
└── Outputs/                    # Çıktı dosyalarının tutulduğu dizin
    ├── final_model.keras       # Eğitilmiş nihai modelin serialize edilmiş hali
    ├── patient_records.db      # SQLite tabanlı yerel klinik kayıt veritabanı (otomatik oluşturulur)
    ├── X_train_pseudo.npy      # Eğitime hazır, temizlenmiş pseudo-label eğitim verisi
    ├── model_karsilastirma_raporu.txt # Mimari kıyaslama performans raporu
    └── *.png                   # ROC, Hata Dağılımı ve diğer analiz grafikleri
```

## Kurulum ve Çalıştırma

Projenin yerel bilgisayarda çalıştırılabilmesi için Python 3.8+ ortamı gerekmektedir. Gerekli bağımlılıkları kurmak ve arayüzü başlatmak için terminal/komut satırında sırasıyla aşağıdaki komutları çalıştırınız:

```bash
# 1. Bağımlılıkların Yüklenmesi
pip install -r requirements.txt

# 2. Klinik Arayüzün Başlatılması
streamlit run app.py
```
Arayüz başlatıldıktan sonra, tarayıcınız otomatik olarak `http://localhost:8501` adresine yönlendirilecektir.

## Model Performans Kıyaslaması
Seçilen 1D-CAE mimarisinin donanımsal verimliliğini doğrulamak adına, karmaşık bir ConvLSTM hibrit modeli ile doğrudan kıyaslama testi uygulanmıştır. Test sonuçlarına göre 1D-CAE, performans kaybı yaşatmadan ciddi bir donanım avantajı sunmaktadır.

| Metrik | 1D-CAE (Baseline) | ConvLSTM Hibrit |
| :--- | :--- | :--- |
| **ROC-AUC Skoru** | 0.9256 | 0.8777 |
| **Toplam Parametre** | 4,826,817 | 3,583,681 |
| **Ortalama Hız (sn/epoch)** | 66.7 sn | 152.2 sn |
| **Normal MSE** | 0.071078 | 0.190060 |
| **AFib MSE** | 0.464752 | 0.721757 |
| **Hata Ayrım Oranı (AFib/Normal)**| 6.54x | 3.80x |

**Teknik Değerlendirme:** 1D-CAE mimarisi, uzun dönemli zamansal belleğe (LSTM) ihtiyaç duymadan 400Hz frekanslı EKG verilerinin yerel morfolojisini üst düzey hassasiyetle modellemeyi başarmıştır. Hesaplama süresinin ConvLSTM'e göre yaklaşık 2.3 kat daha hızlı olması, modeli gerçek zamanlı izleme sistemleri için ideal bir çözüm kılmaktadır.

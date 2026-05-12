# ECG – Atrial Fibrilasyon Tespiti | Proje Teknik Raporu
**Üretken Yapay Zeka Dersi | 1D-Convolutional Autoencoder Tabanlı Anomali Tespiti**

---

## Oluşturulan Dosyalar

| Dosya | Açıklama |
|---|---|
| `ecg_afib_autoencoder.py` | **Ana betik** – Veri yükleme, model, eğitim, görselleştirme, rapor |
| `lstm_autoencoder.py` | Alternatif LSTM-Autoencoder modeli (değiştirilebilir) |
| `sanity_check.py` | Ortam ve model boyutu doğrulayıcı |
| `requirements.txt` | Gerekli Python paketleri |

---

## Model Mimarisi: 1D-CAE

```
Input (4000, 1)
     │
     ▼  ENCODER
Conv1D(32,  k=7, s=2) + BN + LeakyReLU  →  (2000, 32)
Conv1D(64,  k=5, s=2) + BN + LeakyReLU  →  (1000, 64)
Conv1D(128, k=5, s=2) + BN + LeakyReLU  →  (500,  128)
Conv1D(256, k=3, s=2) + BN + LeakyReLU  →  (250,  256)
Conv1D(256, k=3, s=2) + BN + LeakyReLU  →  (125,  256)
Flatten → Dense(64)
     │
  ╔══╧══╗
  ║  64 ║  ← LATENT SPACE (Bottleneck)
  ╚══╤══╝
     │
     ▼  DECODER
Dense(125×256) → Reshape(125, 256)
Conv1DTranspose(256, k=3, s=2) + BN + LeakyReLU  →  (250,  256)
Conv1DTranspose(128, k=3, s=2) + BN + LeakyReLU  →  (500,  128)
Conv1DTranspose(64,  k=5, s=2) + BN + LeakyReLU  →  (1000, 64)
Conv1DTranspose(32,  k=5, s=2) + BN + LeakyReLU  →  (2000, 32)
Conv1DTranspose(1,   k=7, s=2, linear)            →  (4000, 1)
     │
Output (4000, 1)
```

### Katman Özet Tablosu

| Katman | Tip | Çıkış Boyutu | Parametre (~) |
|---|---|---|---|
| ecg_input | InputLayer | (4000, 1) | 0 |
| enc_conv1 | Conv1D | (2000, 32) | 256 |
| enc_conv2 | Conv1D | (1000, 64) | 10.304 |
| enc_conv3 | Conv1D | (500, 128) | 41.088 |
| enc_conv4 | Conv1D | (250, 256) | 98.560 |
| enc_conv5 | Conv1D | (125, 256) | 197.120 |
| enc_flatten | Flatten | (32000,) | 0 |
| **bottleneck** | Dense | **(64,)** | 2.048.064 |
| dec_dense | Dense | (32000,) | 2.080.000 |
| dec_reshape | Reshape | (125, 256) | 0 |
| dec_conv1 | Conv1DTranspose | (250, 256) | 197.120 |
| dec_conv2 | Conv1DTranspose | (500, 128) | 98.432 |
| dec_conv3 | Conv1DTranspose | (1000, 64) | 41.024 |
| dec_conv4 | Conv1DTranspose | (2000, 32) | 10.272 |
| dec_output | Conv1DTranspose | (4000, 1) | 225 |
| **TOPLAM** | | | **~4.8 M** |

> [!NOTE]
> BatchNorm katmanlarının parametreleri yukarıdaki tabloya dahil değildir (her Conv1D'den sonra eklenir, her biri ~2×filtre adet parametre ekler).

---

## Eğitim Stratejisi

```
Eğitim verisi = Normal (500) + Etiketsiz (19.000) = 19.500 sinyal
                          ↓
              %90 Eğitim / %10 Doğrulama bölünmesi
                          ↓
         Autoencoder: Giriş ≡ Hedef  (X → X̂)
                          ↓
              Kayıp Fonksiyonu: MSE
              Optimizer: Adam (lr=1e-3)
              EarlyStopping: patience=10
              ReduceLROnPlateau: factor=0.5, patience=5
```

**AFib sinyalleri EĞİTİME KATILMAZ** → Model sadece "normal" deseni öğrenir.  
Test sırasında AFib sinyalleri yüksek rekonstrüksiyon hatası üretir.

---

## Beklenen Teknik Çıktılar

### Kayıp Değerleri
Eğitim boyunca konsolda her epoch için `Train MSE` ve `Val MSE` yazdırılır.
`Outputs/01_training_history.png` dosyasına grafik olarak kaydedilir.

### Normal vs AFib Hata Farkı
Eğitim bittikten sonra konsolda şu format çıkar:

```
Normal sinyaller – Ortalama MSE  : 0.XXXXXX  (std=0.XXXXXX)
AFib sinyaller   – Ortalama MSE  : 0.XXXXXX  (std=0.XXXXXX)
Hata Farkı Oranı (AFib/Normal)   : X.XXx
Mutlak Ortalama Hata Farkı        : 0.XXXXXX
ROC-AUC Skoru                     : 0.XXXX
```

> [!TIP]
> İyi eğitilmiş bir modelde AFib/Normal hata oranının **3x–10x** arasında olması beklenir. AUC > 0.80 başarılı kabul edilir.

---

## Üretilen Görseller

| Dosya | İçerik |
|---|---|
| `Outputs/01_training_history.png` | Epoch vs MSE kayıp grafiği |
| `Outputs/02_reconstruction_comparison.png` | Normal & AFib orijinal + rekonstrüksiyon + hata |
| `Outputs/03_error_distribution.png` | MSE histogram dağılımı + eşik çizgisi |
| `Outputs/04_roc_curve.png` | ROC eğrisi + AUC skoru |
| `Outputs/05_latent_space_pca.png` | Latent space PCA (2D) görselleştirmesi |
| `Outputs/technical_report.txt` | Tüm metriklerin metin raporu |

---

## Kullanım

```powershell
# 1. Paketleri kur
pip install -r requirements.txt

# 2. Ortam ve model boyutlarını doğrula
python sanity_check.py

# 3. Asıl eğitimi başlat (GPU varsa otomatik kullanılır)
python ecg_afib_autoencoder.py

# 4. Alternatif: LSTM-Autoencoder mimarisini gör
python lstm_autoencoder.py
```

---

## Önemli Parametreler

| Değişken | Varsayılan | Açıklama |
|---|---|---|
| `LATENT_DIM` | 64 | Bottleneck boyutu (32, 128 denenebilir) |
| `EPOCHS` | 50 | EarlyStopping ile erkenden durabilir |
| `BATCH_SIZE` | 32 | GPU varsa 64-128'e çıkarın |
| `LEARNING_RATE` | 1e-3 | Adam optimizer başlangıç hızı |

---

## CSV Format Varsayımı

Kod, iki farklı CSV formatını **otomatik** destekler:
1. **Sadece sinyal**: 4000 sütun (header yok)
2. **İndeks + sinyal**: 4001 sütun (ilk sütun otomatik düşürülür)

> [!WARNING]
> Eğer CSV'nin ilk satırı sütun adı içeriyorsa, `load_and_preprocess()` fonksiyonundaki `header=None` parametresini `header=0` olarak değiştirin.

---

## Proje Mantığı

Bu proje **yarı-denetimli anomali tespiti** prensibiyle çalışır:

1. **Varsayım**: Normal sinüs ritmi (NSR) verilerinde tekrar eden, öğrenilebilir desenler vardır. AFib ise düzensiz, rastgele bir örüntüye sahiptir.

2. **Eğitim**: Autoencoder yalnızca normal + etiketsiz verileri sıkıştırmayı ve yeniden açmayı öğrenir. Normal sinyaller için rekonstrüksiyon hatası (MSE) düşük kalır.

3. **Test**: Görmediği bir AFib sinyali verildiğinde, model bunu "normal deseni" gibi rekonstrükte edemez → **yüksek MSE = anomali**.

4. **Karar**: Belirlenen eşik değerinin üzerindeki MSE → AFib sınıflandırması.

"""
============================================================
Atriyal Fibrilasyon Tespiti - 1D Convolutional Autoencoder
------------------------------------------------------------
Model Architecture : 1D-CAE
Training Paradigm  : Semi-supervised Anomaly Detection
Dataset            : ECG Signal Dataset (400Hz)
============================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # GUI olmayan / terminal ortamlar için
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import tensorflow as tf

# ── Keras importu: tf.keras üzerinden (TF 2.x tüm sürümleriyle uyumlu) ──
from tensorflow.keras import layers, Model, callbacks
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import roc_auc_score, roc_curve, classification_report
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# 0. GLOBAL AYARLAR
# ─────────────────────────────────────────────────────────
DATASET_DIR   = r"c:\Users\aycac\OneDrive\Masaüstü\2026 Bahar\ÜretkenYZ\Proje\Dataset"
TRAIN_CSV     = os.path.join(DATASET_DIR, "traindata.csv")
TEST_CSV      = os.path.join(DATASET_DIR, "testdata.csv")
OUTPUT_DIR    = r"c:\Users\aycac\OneDrive\Masaüstü\2026 Bahar\ÜretkenYZ\Proje\Outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SIGNAL_LENGTH = 4000          # Her sinyaldeki nokta sayısı
BATCH_SIZE    = 32
EPOCHS        = 50
LEARNING_RATE = 1e-3
LATENT_DIM    = 64            # Bottleneck boyutu
RANDOM_SEED   = 42
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────────────────
# 1. VERİ YÜKLEME VE ÖN İŞLEME
# ─────────────────────────────────────────────────────────
def load_and_preprocess(csv_path: str, signal_length: int = 4000,
                         chunksize: int = 500) -> np.ndarray:
    """
    Büyük CSV dosyasını chunk'lar hâlinde okur, Z-score normalleştirir
    ve (N, signal_length, 1) şeklinde döndürür.

    CSV formatı varsayımı:
      - Sütun adı yoksa: her satır bir sinyal, 4000 sütun
        (veya 4001 — ilk sütun indeks ise atlanır)
    """
    print(f"\nINFO: Veri seti yükleniyor -> {csv_path}")
    chunks      = []
    total_rows  = 0

    for chunk in pd.read_csv(csv_path, header=None, chunksize=chunksize,
                              low_memory=True):
        # Eğer ilk sütun tamsayı indeks ise düşür
        if chunk.shape[1] == signal_length + 1:
            chunk = chunk.iloc[:, 1:]
        elif chunk.shape[1] > signal_length:
            chunk = chunk.iloc[:, :signal_length]

        chunk = chunk.astype(np.float32)
        chunks.append(chunk.values)
        total_rows += len(chunk)
        if total_rows % 5000 == 0:
            print(f"INFO: {total_rows} satır okundu.")

    data = np.vstack(chunks)                      # (N, 4000)
    print(f"   ✅ Toplam {data.shape[0]} sinyal, {data.shape[1]} nokta")

    # Z-score normalizasyonu (sinyal başına)
    mean = data.mean(axis=1, keepdims=True)
    std  = data.std(axis=1,  keepdims=True) + 1e-8
    data = (data - mean) / std

    # Reshape → (N, 4000, 1)
    return data[:, :, np.newaxis]


def split_train_data(train_data: np.ndarray):
    """
    Eğitim verisini satır indeksine göre böler.
      - 0-499   → AFib (Kaggle satır 1-500)
      - 500-999 → Normal Sinüs Ritmi (501-1000)
      - 1000+   → Etiketsiz
    """
    afib_data    = train_data[0:500]
    normal_data  = train_data[500:1000]
    unlabel_data = train_data[1000:]

    print(f"\nINFO: Veri Dağılımı")
    print(f"AFib (Etiketli)     : {afib_data.shape[0]} sinyal")
    print(f"Normal (Etiketli)   : {normal_data.shape[0]} sinyal")
    print(f"Etiketsiz           : {unlabel_data.shape[0]} sinyal")
    return afib_data, normal_data, unlabel_data


# ─────────────────────────────────────────────────────────
# 2. MODEL MİMARİSİ: 1D-Convolutional Autoencoder
# ─────────────────────────────────────────────────────────
def build_1d_cae(signal_length: int = 4000, latent_dim: int = 64) -> Model:
    """
    1D-CAE mimarisi:

    ENCODER
    -------
    Input  (4000, 1)
    Conv1D(32,  k=7, stride=2) + BN + LeakyReLU  →  (2000, 32)
    Conv1D(64,  k=5, stride=2) + BN + LeakyReLU  →  (1000, 64)
    Conv1D(128, k=5, stride=2) + BN + LeakyReLU  →  (500,  128)
    Conv1D(256, k=3, stride=2) + BN + LeakyReLU  →  (250,  256)
    Conv1D(256, k=3, stride=2) + BN + LeakyReLU  →  (125,  256)
    Flatten → Dense(latent_dim)  [← BOTTLENECK / LATENT SPACE]

    DECODER
    -------
    Dense(125×256) → Reshape(125, 256)
    Conv1DTranspose(256, k=3, s=2) + BN + LeakyReLU  →  (250,  256)
    Conv1DTranspose(128, k=3, s=2) + BN + LeakyReLU  →  (500,  128)
    Conv1DTranspose(64,  k=5, s=2) + BN + LeakyReLU  →  (1000, 64)
    Conv1DTranspose(32,  k=5, s=2) + BN + LeakyReLU  →  (2000, 32)
    Conv1DTranspose(1,   k=7, s=2, linear)            →  (4000, 1)
    """
    inputs = layers.Input(shape=(signal_length, 1), name="ecg_input")

    # ── ENCODER ──
    x = layers.Conv1D(32,  kernel_size=7, strides=2, padding="same", name="enc_conv1")(inputs)
    x = layers.BatchNormalization(name="enc_bn1")(x)
    x = layers.LeakyReLU(0.2, name="enc_act1")(x)

    x = layers.Conv1D(64,  kernel_size=5, strides=2, padding="same", name="enc_conv2")(x)
    x = layers.BatchNormalization(name="enc_bn2")(x)
    x = layers.LeakyReLU(0.2, name="enc_act2")(x)

    x = layers.Conv1D(128, kernel_size=5, strides=2, padding="same", name="enc_conv3")(x)
    x = layers.BatchNormalization(name="enc_bn3")(x)
    x = layers.LeakyReLU(0.2, name="enc_act3")(x)

    x = layers.Conv1D(256, kernel_size=3, strides=2, padding="same", name="enc_conv4")(x)
    x = layers.BatchNormalization(name="enc_bn4")(x)
    x = layers.LeakyReLU(0.2, name="enc_act4")(x)

    x = layers.Conv1D(256, kernel_size=3, strides=2, padding="same", name="enc_conv5")(x)
    x = layers.BatchNormalization(name="enc_bn5")(x)
    x = layers.LeakyReLU(0.2, name="enc_act5")(x)

    # Boyut kaydediyoruz (flatten öncesi: 125×256)
    enc_len     = signal_length // (2**5)   # 4000 // 32 = 125
    enc_filters = 256

    x      = layers.Flatten(name="enc_flatten")(x)
    latent = layers.Dense(latent_dim, name="bottleneck")(x)   # ← LATENT SPACE

    # ── DECODER ──
    x = layers.Dense(enc_len * enc_filters, name="dec_dense")(latent)
    x = layers.Reshape((enc_len, enc_filters), name="dec_reshape")(x)

    x = layers.Conv1DTranspose(256, kernel_size=3, strides=2, padding="same", name="dec_conv1")(x)
    x = layers.BatchNormalization(name="dec_bn1")(x)
    x = layers.LeakyReLU(0.2, name="dec_act1")(x)

    x = layers.Conv1DTranspose(128, kernel_size=3, strides=2, padding="same", name="dec_conv2")(x)
    x = layers.BatchNormalization(name="dec_bn2")(x)
    x = layers.LeakyReLU(0.2, name="dec_act2")(x)

    x = layers.Conv1DTranspose(64,  kernel_size=5, strides=2, padding="same", name="dec_conv3")(x)
    x = layers.BatchNormalization(name="dec_bn3")(x)
    x = layers.LeakyReLU(0.2, name="dec_act3")(x)

    x = layers.Conv1DTranspose(32,  kernel_size=5, strides=2, padding="same", name="dec_conv4")(x)
    x = layers.BatchNormalization(name="dec_bn4")(x)
    x = layers.LeakyReLU(0.2, name="dec_act4")(x)

    outputs = layers.Conv1DTranspose(1, kernel_size=7, strides=2, padding="same",
                                      activation="linear", name="dec_output")(x)

    model = Model(inputs, outputs, name="1D_CAE")
    return model


# ─────────────────────────────────────────────────────────
# 3. EĞİTİM
# ─────────────────────────────────────────────────────────
def train_model(model: Model,
                train_X: np.ndarray,
                val_X:   np.ndarray,
                epochs:     int   = EPOCHS,
                batch_size: int   = BATCH_SIZE,
                lr:         float = LEARNING_RATE):
    """
    Modeli eğitir.
    - Kayıp fonksiyonu : MSE
    - Optimizer        : Adam (başlangıç lr=1e-3)
    - EarlyStopping    : val_loss ≥ 10 epoch iyileşmezse dur
    - ReduceLROnPlateau: 5 epoch sonra lr yarıya iner
    - ModelCheckpoint  : en iyi val_loss'u kaydeder
    """
    model.compile(optimizer=Adam(learning_rate=lr), loss="mse")

    cb_list = [
        callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                    patience=5, min_lr=1e-6, verbose=1),
    ]

    print(f"\nINFO: Model eğitimi başlatılıyor... (Epochs={epochs}, Batch={batch_size})")
    history = model.fit(
        train_X, train_X,           # Autoencoder: giriş = hedef
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(val_X, val_X),
        callbacks=cb_list,
        verbose=1,
    )
    return history


# ─────────────────────────────────────────────────────────
# 4. ANOMALİ TESPİT FONKSİYONU
# ─────────────────────────────────────────────────────────
def compute_reconstruction_errors(model: Model,
                                   data: np.ndarray,
                                   batch_size: int = 64) -> np.ndarray:
    """
    Her sinyal için MSE (rekonstrüksiyon hatası) hesaplar.
    Yüksek MSE → Anomali (AFib)
    Düşük MSE  → Normal
    """
    reconstructions = model.predict(data, batch_size=batch_size, verbose=0)
    errors = np.mean((data - reconstructions) ** 2, axis=(1, 2))
    return errors


def find_optimal_threshold(normal_errors: np.ndarray,
                            afib_errors:  np.ndarray):
    """
    Youden J istatistiğiyle optimal eşik değerini bulur.
    Döndürür: (threshold, auc, fpr, tpr)
    """
    labels = np.concatenate([np.zeros(len(normal_errors)),
                              np.ones(len(afib_errors))])
    scores = np.concatenate([normal_errors, afib_errors])

    fpr, tpr, thresholds = roc_curve(labels, scores)
    best_idx    = np.argmax(tpr - fpr)
    best_thresh = thresholds[best_idx]
    auc_score   = roc_auc_score(labels, scores)
    return best_thresh, auc_score, fpr, tpr


# ─────────────────────────────────────────────────────────
# 5. GÖRSELLEŞTİRME
# ─────────────────────────────────────────────────────────
DARK_BG  = "#0d1117"
CARD_BG  = "#161b22"
ACCENT1  = "#58a6ff"     # Mavi  (Normal)
ACCENT2  = "#ff7b72"     # Kırmızı (AFib)
ACCENT3  = "#3fb950"     # Yeşil (Rekonstrüksiyon)
TEXT_CLR = "#e6edf3"
GRID_CLR = "#30363d"

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor":   CARD_BG,
    "axes.edgecolor":   GRID_CLR,
    "axes.labelcolor":  TEXT_CLR,
    "xtick.color":      TEXT_CLR,
    "ytick.color":      TEXT_CLR,
    "text.color":       TEXT_CLR,
    "grid.color":       GRID_CLR,
    "grid.alpha":       0.5,
    "font.family":      "DejaVu Sans",
    "font.size":        10,
})


def plot_reconstruction_comparison(model, normal_signal, afib_signal, save_path: str):
    """Normal ve AFib sinyallerini orijinal + rekonstrüksiyon + hata olarak karşılaştırır."""
    norm_recon = model.predict(normal_signal[np.newaxis], verbose=0)[0, :, 0]
    afib_recon = model.predict(afib_signal[np.newaxis],  verbose=0)[0, :, 0]

    norm_orig  = normal_signal[:, 0]
    afib_orig  = afib_signal[:, 0]
    t          = np.arange(SIGNAL_LENGTH) / 400.0    # zaman ekseni (saniye)

    norm_mse = np.mean((norm_orig - norm_recon) ** 2)
    afib_mse = np.mean((afib_orig - afib_recon) ** 2)

    fig = plt.figure(figsize=(18, 10), facecolor=DARK_BG)
    fig.suptitle("ECG Rekonstrüksiyon Karşılaştırması: Normal vs AFib",
                 fontsize=16, fontweight="bold", color=TEXT_CLR, y=0.98)
    gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, norm_orig,  color=ACCENT1, alpha=0.9, lw=1.0, label="Orijinal")
    ax1.plot(t, norm_recon, color=ACCENT3, alpha=0.85, lw=0.8, ls="--", label="Rekonstrüksiyon")
    ax1.set_title(f"Normal Sinyal  |  MSE = {norm_mse:.5f}", fontsize=11)
    ax1.set_xlabel("Zaman (s)"); ax1.set_ylabel("Amplitude (Z-score)")
    ax1.legend(loc="upper right"); ax1.grid(True)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, afib_orig,  color=ACCENT2, alpha=0.9, lw=1.0, label="Orijinal")
    ax2.plot(t, afib_recon, color=ACCENT3, alpha=0.85, lw=0.8, ls="--", label="Rekonstrüksiyon")
    ax2.set_title(f"AFib Sinyal  |  MSE = {afib_mse:.5f}", fontsize=11)
    ax2.set_xlabel("Zaman (s)"); ax2.set_ylabel("Amplitude (Z-score)")
    ax2.legend(loc="upper right"); ax2.grid(True)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.fill_between(t, np.abs(norm_orig - norm_recon), alpha=0.6, color=ACCENT1,
                     label=f"Abs. Hata (MSE={norm_mse:.5f})")
    ax3.set_title("Normal Sinyal – Rekonstrüksiyon Hatası", fontsize=11)
    ax3.set_xlabel("Zaman (s)"); ax3.set_ylabel("|Orijinal - Rekon.|")
    ax3.legend(); ax3.grid(True)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.fill_between(t, np.abs(afib_orig - afib_recon), alpha=0.6, color=ACCENT2,
                     label=f"Abs. Hata (MSE={afib_mse:.5f})")
    ax4.set_title("AFib Sinyal – Rekonstrüksiyon Hatası", fontsize=11)
    ax4.set_xlabel("Zaman (s)"); ax4.set_ylabel("|Orijinal - Rekon.|")
    ax4.legend(); ax4.grid(True)

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    print(f"INFO: Grafik kaydedildi -> {save_path}")
    plt.close()


def plot_training_history(history, save_path: str):
    """Epoch vs MSE kayıp grafiği."""
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=DARK_BG)
    ep = range(1, len(history.history["loss"]) + 1)
    ax.plot(ep, history.history["loss"],     color=ACCENT1, lw=2, label="Egitim Kaybi (MSE)")
    ax.plot(ep, history.history["val_loss"], color=ACCENT2, lw=2, ls="--", label="Dogrulama Kaybi (MSE)")
    ax.set_title("Egitim Surecinde Kayip Degisimi", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE")
    ax.legend(); ax.grid(True)
    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    print(f"INFO: Grafik kaydedildi -> {save_path}")
    plt.close()


def plot_error_distribution(normal_errors, afib_errors, threshold, save_path: str):
    """MSE histogram dağılımı + eşik çizgisi."""
    fig, ax = plt.subplots(figsize=(11, 5), facecolor=DARK_BG)
    ax.hist(normal_errors, bins=60, alpha=0.70, color=ACCENT1, density=True,
            label=f"Normal  (n={len(normal_errors)}, ort.={normal_errors.mean():.5f})")
    ax.hist(afib_errors,   bins=60, alpha=0.70, color=ACCENT2, density=True,
            label=f"AFib    (n={len(afib_errors)}, ort.={afib_errors.mean():.5f})")
    ax.axvline(threshold, color="#ffa657", lw=2, ls="--",
               label=f"Esik Degeri = {threshold:.5f}")
    ax.set_title("Rekonstruksiyon Hatasi (MSE) Dagilimi", fontsize=14, fontweight="bold")
    ax.set_xlabel("MSE"); ax.set_ylabel("Yogunluk")
    ax.legend(); ax.grid(True)
    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    print(f"INFO: Grafik kaydedildi -> {save_path}")
    plt.close()


def plot_roc_curve(fpr, tpr, auc_score, save_path: str):
    """ROC eğrisi."""
    fig, ax = plt.subplots(figsize=(7, 6), facecolor=DARK_BG)
    ax.plot(fpr, tpr, color=ACCENT1, lw=2, label=f"ROC (AUC = {auc_score:.4f})")
    ax.plot([0, 1], [0, 1], color=GRID_CLR, ls="--", lw=1)
    ax.fill_between(fpr, tpr, alpha=0.15, color=ACCENT1)
    ax.set_title("ROC Egrisi - AFib Anomali Tespiti", fontsize=13, fontweight="bold")
    ax.set_xlabel("Yanlis Pozitif Orani (FPR)")
    ax.set_ylabel("Dogru Pozitif Orani (TPR)")
    ax.legend(); ax.grid(True)
    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    print(f"INFO: Grafik kaydedildi -> {save_path}")
    plt.close()


def plot_latent_space(model, normal_data, afib_data, save_path: str):
    """Latent vektörleri PCA ile 2D'ye indirip görselleştirir."""
    from sklearn.decomposition import PCA

    encoder = Model(inputs=model.input,
                    outputs=model.get_layer("bottleneck").output)

    n_sample = min(200, len(normal_data), len(afib_data))
    idx_n    = np.random.choice(len(normal_data), n_sample, replace=False)
    idx_a    = np.random.choice(len(afib_data),   n_sample, replace=False)

    z_normal = encoder.predict(normal_data[idx_n], verbose=0)
    z_afib   = encoder.predict(afib_data[idx_a],   verbose=0)

    pca   = PCA(n_components=2, random_state=RANDOM_SEED)
    z_all = pca.fit_transform(np.vstack([z_normal, z_afib]))
    z_n, z_a = z_all[:n_sample], z_all[n_sample:]
    var = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(8, 7), facecolor=DARK_BG)
    ax.scatter(z_n[:, 0], z_n[:, 1], c=ACCENT1, alpha=0.75, s=35, edgecolors="none", label="Normal")
    ax.scatter(z_a[:, 0], z_a[:, 1], c=ACCENT2, alpha=0.75, s=35, edgecolors="none", label="AFib")
    ax.set_title("Latent Space - PCA (2D)", fontsize=13, fontweight="bold")
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}%)")
    ax.legend(); ax.grid(True)
    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    print(f"INFO: Grafik kaydedildi -> {save_path}")
    plt.close()


# ─────────────────────────────────────────────────────────
# 6. TEKNİK RAPOR ÇIKTISI
# ─────────────────────────────────────────────────────────
def print_technical_report(model, history,
                            normal_errors, afib_errors,
                            auc_score, threshold):
    sep = "=" * 65
    print(f"\n{sep}")
    print("  TEKNIK RAPOR - Model Ozeti ve Sonuclar")
    print(sep)

    # Katman detayları
    print(f"\n  Model Adi        : {model.name}")
    print(f"  Toplam Parametre : {model.count_params():,}")
    print()
    print(f"  {'Katman Adi':<28} {'Tipi':<30} {'Parametre'}")
    print(f"  {'-'*28} {'-'*30} {'-'*12}")
    for layer in model.layers:
        try:
            n_p = layer.count_params()
        except Exception:
            n_p = 0
        print(f"  {layer.name:<28} {type(layer).__name__:<30} {n_p:,}")

    # Eğitim kaybı
    print(f"\n  EGITIM KAYIP DEGERLERI (her 5 epoch):")
    train_loss = history.history["loss"]
    val_loss   = history.history["val_loss"]
    print(f"  {'Epoch':<8} {'Train MSE':<15} {'Val MSE'}")
    print(f"  {'-'*8} {'-'*15} {'-'*12}")
    for i, (tl, vl) in enumerate(zip(train_loss, val_loss)):
        if i % 5 == 0 or i == len(train_loss) - 1:
            print(f"  {i+1:<8} {tl:<15.6f} {vl:.6f}")

    # Anomali metrikleri
    print(f"\n  ANOMALI TESPIT SONUCLARI:")
    print(f"  Normal sinyaller - Ort. MSE : {normal_errors.mean():.6f}  (std={normal_errors.std():.6f})")
    print(f"  AFib sinyaller   - Ort. MSE : {afib_errors.mean():.6f}  (std={afib_errors.std():.6f})")
    ratio = afib_errors.mean() / (normal_errors.mean() + 1e-10)
    delta = afib_errors.mean() - normal_errors.mean()
    print(f"  Hata Farki Orani (AFib/Normal) : {ratio:.2f}x")
    print(f"  Mutlak Ortalama Hata Farki     : {delta:.6f}")
    print(f"  Optimal Esik Degeri            : {threshold:.6f}  (Youden J)")
    print(f"  ROC-AUC Skoru                  : {auc_score:.4f}")

    # Sınıflandırma raporu
    labels = np.array([0]*len(normal_errors) + [1]*len(afib_errors))
    scores = np.concatenate([normal_errors, afib_errors])
    preds  = (scores >= threshold).astype(int)
    print(f"\n  SINIFLANDIRMA RAPORU (esik={threshold:.5f}):")
    print(classification_report(labels, preds,
                                 target_names=["Normal", "AFib"], digits=4))
    print(sep)

    # Dosyaya kaydet
    report_path = os.path.join(OUTPUT_DIR, "technical_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("ECG AFib Autoencoder - Teknik Rapor\n")
        f.write(sep + "\n")
        f.write(f"Model: {model.name}\n")
        f.write(f"Toplam Parametre: {model.count_params():,}\n\n")
        f.write("Egitim Kayip Degerleri:\n")
        f.write(f"{'Epoch':<8} {'Train MSE':<15} {'Val MSE'}\n")
        for i, (tl, vl) in enumerate(zip(train_loss, val_loss)):
            f.write(f"{i+1:<8} {tl:<15.6f} {vl:.6f}\n")
        f.write(f"\nNormal MSE (ort.): {normal_errors.mean():.6f}\n")
        f.write(f"AFib   MSE (ort.): {afib_errors.mean():.6f}\n")
        f.write(f"Hata Orani       : {ratio:.2f}x\n")
        f.write(f"ROC-AUC          : {auc_score:.4f}\n")
        f.write(f"Esik Degeri      : {threshold:.6f}\n")
    print(f"\n  Rapor kaydedildi: {report_path}")


# ─────────────────────────────────────────────────────────
# 7. ANA AKIŞ
# ─────────────────────────────────────────────────────────
def main():
    print("\n" + "="*65)
    print("  ECG - Atrial Fibrilasyon Tespiti (1D-CAE)")
    print("="*65)
    print(f"  TensorFlow : {tf.__version__}")
    gpus = tf.config.list_physical_devices("GPU")
    print(f"  GPU        : {gpus if gpus else 'Yok (CPU)'}")

    # ── 1. Veri Yükleme ──
    train_all = load_and_preprocess(TRAIN_CSV, SIGNAL_LENGTH)
    afib_data, normal_data, unlabel_data = split_train_data(train_all)

    # ── 2. Eğitim Seti: Normal + Etiketsiz ──
    train_set = np.concatenate([normal_data, unlabel_data], axis=0)
    idx       = np.random.permutation(len(train_set))
    train_set = train_set[idx]

    split   = int(0.9 * len(train_set))
    train_X = train_set[:split]
    val_X   = train_set[split:]
    print(f"\n  Egitim  : {train_X.shape}")
    print(f"  Val     : {val_X.shape}")

    # ── 3. Model ──
    model = build_1d_cae(SIGNAL_LENGTH, LATENT_DIM)
    model.summary()

    # ── 4. Eğitim ──
    history = train_model(model, train_X, val_X)
    # Not: EarlyStopping(restore_best_weights=True) en iyi agirliklari
    # otomatik olarak hafizada tutar, ayrica dosyadan yuklemeye gerek yok.

    # ── 5. Rekonstrüksiyon Hataları ──
    print("\n  Normal sinyaller degerlendiriliyor...")
    normal_errors = compute_reconstruction_errors(model, normal_data)
    print("  AFib sinyaller degerlendiriliyor...")
    afib_errors   = compute_reconstruction_errors(model, afib_data)

    # ── 6. Eşik & ROC ──
    threshold, auc_score, fpr, tpr = find_optimal_threshold(normal_errors, afib_errors)

    # ── 7. Görseller ──
    print("\n  Gorseller olusturuluyor...")
    plot_training_history(history,
        save_path=os.path.join(OUTPUT_DIR, "01_training_history.png"))

    n_idx = np.random.randint(0, len(normal_data))
    a_idx = np.random.randint(0, len(afib_data))
    plot_reconstruction_comparison(model,
        normal_signal=normal_data[n_idx],
        afib_signal=afib_data[a_idx],
        save_path=os.path.join(OUTPUT_DIR, "02_reconstruction_comparison.png"))

    plot_error_distribution(normal_errors, afib_errors, threshold,
        save_path=os.path.join(OUTPUT_DIR, "03_error_distribution.png"))

    plot_roc_curve(fpr, tpr, auc_score,
        save_path=os.path.join(OUTPUT_DIR, "04_roc_curve.png"))

    plot_latent_space(model, normal_data, afib_data,
        save_path=os.path.join(OUTPUT_DIR, "05_latent_space_pca.png"))

    # ── 8. Teknik Rapor ──
    print_technical_report(model, history, normal_errors, afib_errors,
                            auc_score, threshold)

    print(f"\n  Tum ciktilar: '{OUTPUT_DIR}'")
    print("="*65 + "\n")


if __name__ == "__main__":
    main()

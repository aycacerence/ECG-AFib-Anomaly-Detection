"""
============================================================
Hiperparametre Optimizasyonu — Grid Search
ECG AFib Tespiti | 1D-CAE
============================================================

Test edilen kombinasyonlar (6 adet):
  Latent Dim  :  32,  64,  128
  Learning Rate:  1e-3,  1e-4

Giriş:
  Outputs/X_train_pseudo.npy   (4000, 4000, 1) — pseudo-label pipeline çıktısı
  Dataset/traindata.csv        satır 0-499 AFib, satır 500-999 Normal

Çıktı:
  Outputs/hiperparametre_sonuclari.csv
  Outputs/08_grid_search_summary.png
  Konsola EN İYİ MODEL bilgisi
============================================================
"""

import os, time, gc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_auc_score, roc_curve
import warnings
warnings.filterwarnings("ignore")

# ─── YOLLAR ─────────────────────────────────────────────
DATASET_DIR = r"c:\Users\aycac\OneDrive\Masaüstü\2026 Bahar\ÜretkenYZ\Proje\Dataset"
TRAIN_CSV   = os.path.join(DATASET_DIR, "traindata.csv")
OUTPUT_DIR  = r"c:\Users\aycac\OneDrive\Masaüstü\2026 Bahar\ÜretkenYZ\Proje\Outputs"
XTRAIN_NPY  = os.path.join(OUTPUT_DIR, "X_train_pseudo.npy")
RESULTS_CSV = os.path.join(OUTPUT_DIR, "hiperparametre_sonuclari.csv")
PLOT_PATH   = os.path.join(OUTPUT_DIR, "08_grid_search_summary.png")

SIGNAL_LENGTH = 4000
BATCH_SIZE    = 32
MAX_EPOCHS    = 20
PATIENCE      = 7
SEED          = 42

# ─── ARANACAK PARAMETRELER ──────────────────────────────
LATENT_DIMS    = [32, 64, 128]
LEARNING_RATES = [1e-3, 1e-4]


# ─────────────────────────────────────────────────────────
# MODEL — aynı 5 katmanlı 1D-CAE (ecg_afib_autoencoder.py'den)
# ─────────────────────────────────────────────────────────
def build_1d_cae(latent_dim: int) -> Model:
    inp = layers.Input(shape=(SIGNAL_LENGTH, 1))

    x = layers.Conv1D(32,  7, 2, "same")(inp)
    x = layers.BatchNormalization()(x); x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv1D(64,  5, 2, "same")(x)
    x = layers.BatchNormalization()(x); x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv1D(128, 5, 2, "same")(x)
    x = layers.BatchNormalization()(x); x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv1D(256, 3, 2, "same")(x)
    x = layers.BatchNormalization()(x); x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv1D(256, 3, 2, "same")(x)
    x = layers.BatchNormalization()(x); x = layers.LeakyReLU(0.2)(x)

    x      = layers.Flatten()(x)
    latent = layers.Dense(latent_dim, name="bottleneck")(x)

    enc_len = SIGNAL_LENGTH // 32        # 125
    x = layers.Dense(enc_len * 256)(latent)
    x = layers.Reshape((enc_len, 256))(x)

    x = layers.Conv1DTranspose(256, 3, 2, "same")(x)
    x = layers.BatchNormalization()(x); x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv1DTranspose(128, 3, 2, "same")(x)
    x = layers.BatchNormalization()(x); x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv1DTranspose(64,  5, 2, "same")(x)
    x = layers.BatchNormalization()(x); x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv1DTranspose(32,  5, 2, "same")(x)
    x = layers.BatchNormalization()(x); x = layers.LeakyReLU(0.2)(x)
    out = layers.Conv1DTranspose(1,  7, 2, "same", activation="linear")(x)

    return Model(inp, out, name=f"CAE_ld{latent_dim}")


# ─────────────────────────────────────────────────────────
# VERİ YÜKLEME
# ─────────────────────────────────────────────────────────
def load_labeled_from_csv() -> tuple:
    """traindata.csv'den AFib (0-499) ve Normal (500-999) yükler + Z-score."""
    print("  Etiketli veriler yukleniyor (AFib + Normal)...")
    chunks, cursor = [], 0
    for ch in pd.read_csv(TRAIN_CSV, header=None, chunksize=500, low_memory=True):
        if cursor >= 1000:
            break
        end = min(cursor + len(ch), 1000)
        local_end = end - cursor
        arr = ch.iloc[:local_end].values.astype(np.float32)
        if arr.shape[1] == SIGNAL_LENGTH + 1:
            arr = arr[:, 1:]
        elif arr.shape[1] > SIGNAL_LENGTH:
            arr = arr[:, :SIGNAL_LENGTH]
        chunks.append(arr)
        cursor += len(ch)

    labeled = np.vstack(chunks)[:1000]  # (1000, 4000)
    # Z-score (sinyal basina)
    mu  = labeled.mean(axis=1, keepdims=True)
    sig = labeled.std(axis=1,  keepdims=True) + 1e-8
    labeled = (labeled - mu) / sig

    afib   = labeled[:500,  :, np.newaxis]   # (500, 4000, 1)
    normal = labeled[500:1000, :, np.newaxis] # (500, 4000, 1)
    print(f"    AFib  : {afib.shape}")
    print(f"    Normal: {normal.shape}")
    return afib, normal


def compute_mse(model, data):
    """Sinyal basina ortalama MSE."""
    recon = model.predict(data, batch_size=64, verbose=0)
    return np.mean((data - recon) ** 2, axis=(1, 2))


def evaluate_model(model, afib, normal):
    """ROC-AUC, ortalama MSE'ler ve Youden-J esigini hesaplar."""
    err_n = compute_mse(model, normal)
    err_a = compute_mse(model, afib)

    labels = np.concatenate([np.zeros(len(err_n)), np.ones(len(err_a))])
    scores = np.concatenate([err_n, err_a])

    auc = roc_auc_score(labels, scores)
    fpr, tpr, ths = roc_curve(labels, scores)
    best = ths[np.argmax(tpr - fpr)]

    return {
        "normal_mse_mean": float(err_n.mean()),
        "afib_mse_mean":   float(err_a.mean()),
        "mse_ratio":       float(err_a.mean() / (err_n.mean() + 1e-10)),
        "roc_auc":         float(auc),
        "threshold":       float(best),
    }


# ─────────────────────────────────────────────────────────
# GRID SEARCH
# ─────────────────────────────────────────────────────────
def run_grid_search():
    print("\n" + "="*60)
    print("  HIPERPARAMETRE GRID SEARCH")
    print("="*60)

    # ── Veri ──
    print("\n[1/3] Veriler hazirlaniyor...")
    X_train = np.load(XTRAIN_NPY)
    print(f"  X_train (pseudo): {X_train.shape}")

    afib_test, normal_test = load_labeled_from_csv()

    # %90 egitim / %10 dogrulama
    np.random.seed(SEED)
    perm    = np.random.permutation(len(X_train))
    X_train = X_train[perm]
    split   = int(0.9 * len(X_train))
    trn, val = X_train[:split], X_train[split:]
    print(f"  Egitim : {trn.shape}")
    print(f"  Val    : {val.shape}")

    # ── Grid ──
    combos = [(ld, lr) for ld in LATENT_DIMS for lr in LEARNING_RATES]
    print(f"\n[2/3] {len(combos)} kombinasyon test edilecek:")
    for i, (ld, lr) in enumerate(combos, 1):
        print(f"  {i}. latent_dim={ld:>4d}  lr={lr}")

    results = []

    for idx, (ld, lr) in enumerate(combos, 1):
        print(f"\n{'─'*60}")
        print(f"  [{idx}/{len(combos)}]  latent_dim={ld}  lr={lr}")
        print(f"{'─'*60}")

        tf.random.set_seed(SEED)
        np.random.seed(SEED)

        model = build_1d_cae(ld)
        model.compile(optimizer=Adam(learning_rate=lr), loss="mse")

        t0 = time.time()
        hist = model.fit(
            trn, trn,
            epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(val, val),
            callbacks=[
                callbacks.EarlyStopping(monitor="val_loss",
                                        patience=PATIENCE,
                                        restore_best_weights=True,
                                        verbose=1),
            ],
            verbose=1,
        )
        elapsed = time.time() - t0
        epochs_ran = len(hist.history["loss"])

        # Degerlendirme
        metrics = evaluate_model(model, afib_test, normal_test)
        final_train_loss = hist.history["loss"][-1]
        final_val_loss   = hist.history["val_loss"][-1]
        best_val_loss    = min(hist.history["val_loss"])

        row = {
            "latent_dim":     ld,
            "learning_rate":  lr,
            "epochs_ran":     epochs_ran,
            "train_loss":     round(final_train_loss, 6),
            "val_loss":       round(final_val_loss, 6),
            "best_val_loss":  round(best_val_loss, 6),
            "normal_mse":     round(metrics["normal_mse_mean"], 6),
            "afib_mse":       round(metrics["afib_mse_mean"], 6),
            "mse_ratio":      round(metrics["mse_ratio"], 2),
            "roc_auc":        round(metrics["roc_auc"], 4),
            "threshold":      round(metrics["threshold"], 6),
            "params":         model.count_params(),
            "time_sec":       round(elapsed, 1),
        }
        results.append(row)

        print(f"\n  Sonuc: AUC={row['roc_auc']:.4f}  |  "
              f"MSE orani={row['mse_ratio']:.2f}x  |  "
              f"Val loss={row['best_val_loss']:.6f}  |  "
              f"{elapsed:.0f}s ({epochs_ran} epoch)")

        # Bellek temizle
        del model
        tf.keras.backend.clear_session()
        gc.collect()

    # ── Sonuçları kaydet ──
    print(f"\n[3/3] Sonuclar kaydediliyor...")
    df = pd.DataFrame(results)
    df = df.sort_values("roc_auc", ascending=False).reset_index(drop=True)
    df.index.name = "sira"
    df.to_csv(RESULTS_CSV, index=True)
    print(f"\n  CSV: {RESULTS_CSV}")

    # Tablo
    print("\n" + "="*60)
    print("  GRID SEARCH SONUCLARI")
    print("="*60)
    print(df.to_string(index=True))

    # En iyi model
    best = df.iloc[0]
    print(f"\n{'='*60}")
    print(f"  EN IYI MODEL")
    print(f"{'='*60}")
    print(f"  Latent Dim    : {int(best['latent_dim'])}")
    print(f"  Learning Rate : {best['learning_rate']}")
    print(f"  ROC-AUC       : {best['roc_auc']:.4f}")
    print(f"  MSE Orani     : {best['mse_ratio']:.2f}x")
    print(f"  Normal MSE    : {best['normal_mse']:.6f}")
    print(f"  AFib MSE      : {best['afib_mse']:.6f}")
    print(f"  Esik Degeri   : {best['threshold']:.6f}")
    print(f"  Epoch         : {int(best['epochs_ran'])}")
    print(f"  Parametre     : {int(best['params']):,}")
    print(f"{'='*60}")

    # ── Görsel ──
    _plot_results(df)

    return df


# ─────────────────────────────────────────────────────────
# GÖRSELLEŞTİRME
# ─────────────────────────────────────────────────────────
DARK_BG = "#0d1117"; CARD_BG = "#161b22"
A1 = "#58a6ff"; A2 = "#ff7b72"; A3 = "#3fb950"; TX = "#e6edf3"

plt.rcParams.update({
    "figure.facecolor": DARK_BG, "axes.facecolor": CARD_BG,
    "axes.edgecolor": "#30363d", "axes.labelcolor": TX,
    "xtick.color": TX, "ytick.color": TX, "text.color": TX,
    "grid.color": "#30363d", "grid.alpha": 0.5, "font.size": 9,
})


def _plot_results(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), facecolor=DARK_BG)
    fig.suptitle("Hiperparametre Grid Search Sonuclari",
                 fontsize=14, fontweight="bold", color=TX)

    labels = [f"ld={int(r['latent_dim'])}\nlr={r['learning_rate']}"
              for _, r in df.iterrows()]
    x_pos = np.arange(len(df))

    # (a) ROC-AUC barchart
    ax = axes[0]
    colors = [A3 if i == 0 else A1 for i in range(len(df))]
    bars = ax.bar(x_pos, df["roc_auc"], color=colors, alpha=0.85)
    for bar, val in zip(bars, df["roc_auc"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8, color=TX)
    ax.set_xticks(x_pos); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("ROC-AUC"); ax.set_title("ROC-AUC Karsilastirmasi")
    ax.set_ylim(0.5, 1.0); ax.grid(True, axis="y")

    # (b) MSE oranı
    ax = axes[1]
    bars = ax.bar(x_pos, df["mse_ratio"], color=[A2]*len(df), alpha=0.85)
    for bar, val in zip(bars, df["mse_ratio"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{val:.2f}x", ha="center", va="bottom", fontsize=8, color=TX)
    ax.set_xticks(x_pos); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("AFib MSE / Normal MSE")
    ax.set_title("MSE Ayrim Gucu"); ax.grid(True, axis="y")

    # (c) Eğitim süresi
    ax = axes[2]
    bars = ax.bar(x_pos, df["time_sec"], color=["#d2a8ff"]*len(df), alpha=0.85)
    for bar, val in zip(bars, df["time_sec"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{val:.0f}s", ha="center", va="bottom", fontsize=8, color=TX)
    ax.set_xticks(x_pos); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Saniye"); ax.set_title("Egitim Suresi")
    ax.grid(True, axis="y")

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    print(f"  Gorsel: {PLOT_PATH}")
    plt.close()


# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    results_df = run_grid_search()
    print("\nGrid search tamamlandi.\n")

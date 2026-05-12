"""
============================================================
Pseudo-labeling Data Pipeline — ECG AFib Tespiti Projesi
============================================================
Aşamalar:
  1. Normal sinyalleri (satır 500-999) yükle → Z-score → referans şablon
  2. Etiketsiz (1000-19999) sinyalleri CHUNK okuyarak:
     a) Z-score normalizasyonu
     b) İstatistiksel filtreler (genlik, düz-çizgi, ZCR)
     c) Normal şablonla Pearson korelasyonu hesapla
     → Yalnızca (row_idx, score) çifti sakla [bellek dostu]
  3. En yüksek skorlu TOP_K sinyali seç
  4. İkinci geçişte sadece seçilen satırları çek
  5. 500 etiketli Normal + TOP_K pseudo-Normal → X_train.npy
============================================================
"""

import os, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ─── YAPILANDIRMA ───────────────────────────────────────
DATASET_DIR  = r"c:\Users\aycac\OneDrive\Masaüstü\2026 Bahar\ÜretkenYZ\Proje\Dataset"
TRAIN_CSV    = os.path.join(DATASET_DIR, "traindata.csv")
OUTPUT_DIR   = r"c:\Users\aycac\OneDrive\Masaüstü\2026 Bahar\ÜretkenYZ\Proje\Outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

XTRAIN_NPY   = os.path.join(OUTPUT_DIR, "X_train_pseudo.npy")
REPORT_TXT   = os.path.join(OUTPUT_DIR, "pseudo_label_report.txt")
PLOT_PATH    = os.path.join(OUTPUT_DIR, "07_pseudo_label_summary.png")

SIGNAL_LENGTH   = 4000
CHUNK_SIZE      = 500
AFIB_END        = 500      # satır 0-499
NORMAL_END      = 1000     # satır 500-999
UNLABELED_START = 1000     # satır 1000-19999

TOP_K           = 3500     # kaç pseudo-normal seçilecek

# ── Filtre eşikleri ──
AMP_MAX   = 1500.0   # |max| µV
MIN_STD   = 0.05     # düz çizgi alt sınırı
MAX_ZCR   = 0.45     # sıfır-geçiş oranı üst sınırı
IQR_MULT  = 5.0      # enerji IQR çarpanı

# ─────────────────────────────────────────────────────────
# YARDIMCI FONKSİYONLAR
# ─────────────────────────────────────────────────────────
def _fix_cols(df: pd.DataFrame) -> np.ndarray:
    """Sütun sayısını düzelt ve float32 ndarray döndür."""
    if df.shape[1] == SIGNAL_LENGTH + 1:
        df = df.iloc[:, 1:]
    elif df.shape[1] > SIGNAL_LENGTH:
        df = df.iloc[:, :SIGNAL_LENGTH]
    return df.astype(np.float32).values


def zscore(signals: np.ndarray) -> np.ndarray:
    """Her sinyal kendi içinde Z-score normalize edilir."""
    mu  = signals.mean(axis=1, keepdims=True)
    sig = signals.std(axis=1,  keepdims=True) + 1e-8
    return (signals - mu) / sig


def stat_filter_mask(signals: np.ndarray) -> np.ndarray:
    """
    True → geçer, False → elenir
    Filtreler: genlik | düz-çizgi | ZCR
    Not: Enerji IQR filtresi global istatistik gerektirir;
         buraya ham verilere uygulanır.
    """
    m_amp  = np.max(np.abs(signals), axis=1) <= AMP_MAX
    m_flat = signals.std(axis=1) >= MIN_STD
    centered   = signals - signals.mean(axis=1, keepdims=True)
    zcr        = (np.diff(np.sign(centered), axis=1) != 0).sum(axis=1) / SIGNAL_LENGTH
    m_zcr      = zcr <= MAX_ZCR
    return m_amp & m_flat & m_zcr


def pearson_corr_with_template(signals: np.ndarray,
                                template: np.ndarray) -> np.ndarray:
    """
    Her sinyalin referans şablonla Pearson korelasyonunu hesapla.
    signals  : (N, L) — Z-score uygulanmış
    template : (L,)   — referans normal şablon (Z-score ortalama)
    Döndürür : (N,) korelasyon katsayıları [-1, 1]
    """
    # Z-score sonrası her ikisi de sıfır-ortalama, birim-varyans (yaklaşık)
    t = template - template.mean()
    t_norm = np.linalg.norm(t) + 1e-8
    scores = np.empty(len(signals), dtype=np.float32)
    for i, s in enumerate(signals):
        s2 = s - s.mean()
        scores[i] = np.dot(s2, t) / (np.linalg.norm(s2) * t_norm + 1e-8)
    return scores


# ─────────────────────────────────────────────────────────
# ADIM 1 — Etiketli Normal Sinyalleri Yükle & Şablon Oluştur
# ─────────────────────────────────────────────────────────
def load_labeled_normal() -> np.ndarray:
    """
    traindata.csv'den satır 500-999'u okur.
    Döndürür: Z-score normalize edilmiş (500, 4000) normal sinyaller.
    """
    print("\n[1/4] Etiketli Normal sinyaller yükleniyor (satır 500-999)...")
    rows_collected = []
    row_cursor     = 0

    for chunk_df in pd.read_csv(TRAIN_CSV, header=None,
                                 chunksize=CHUNK_SIZE, low_memory=True):
        chunk_start = row_cursor
        chunk_end   = row_cursor + len(chunk_df)

        # Normal aralıkla kesişimi bul
        s = max(chunk_start, AFIB_END)
        e = min(chunk_end,   NORMAL_END)
        if s < e:
            local_s = s - chunk_start
            local_e = e - chunk_start
            rows_collected.append(_fix_cols(chunk_df.iloc[local_s:local_e]))

        row_cursor = chunk_end
        if row_cursor >= NORMAL_END:
            break   # ihtiyaç olan satırları aldık

    normal_raw = np.vstack(rows_collected)               # (500, 4000)
    normal_z   = zscore(normal_raw)                      # Z-score
    print(f"   Normal sinyal: {normal_z.shape} | "
          f"mean={normal_z.mean():.3f}, std={normal_z.std():.3f}")
    return normal_z


def build_template(normal_z: np.ndarray) -> np.ndarray:
    """Referans normal şablonu: 500 Z-score sinyalinin ortalaması."""
    template = normal_z.mean(axis=0)   # (4000,)
    print(f"   Referans şablon oluşturuldu — şekil: {template.shape}")
    return template


# ─────────────────────────────────────────────────────────
# ADIM 2 — Etiketsiz Veriyi Tara (1. Geçiş, bellek dostu)
# ─────────────────────────────────────────────────────────
def pass1_score_unlabeled(template: np.ndarray) -> tuple:
    """
    Etiketsiz sinyalleri chunk'lar hâlinde okur.
    Her sinyal için:  filtre → Z-score → korelasyon skoru
    Yalnızca (global_row_idx, score) çifti saklanır → çok küçük bellek.

    Döndürür:
        indices (np.ndarray int32) : global CSV satır indeksleri
        scores  (np.ndarray float32): Pearson korelasyon skorları
        filter_stats (dict)
    """
    print("\n[2/4] Etiketsiz sinyaller taranıyor (1. geçiş)...")
    t0         = time.time()
    row_cursor = 0
    chunk_idx  = 0

    all_indices = []
    all_scores  = []
    all_energies= []
    stats = {"okunan": 0, "amp_elendi": 0,
             "flat_elendi": 0, "zcr_elendi": 0, "filtre_gecen": 0}

    for chunk_df in pd.read_csv(TRAIN_CSV, header=None,
                                 chunksize=CHUNK_SIZE, low_memory=True):

        chunk_end = row_cursor + len(chunk_df)

        # Etiketsiz aralıkla kesişim
        s = max(row_cursor, UNLABELED_START)
        e = chunk_end
        if s >= e:
            row_cursor = chunk_end
            continue

        local_s = s - row_cursor
        raw     = _fix_cols(chunk_df.iloc[local_s:])

        stats["okunan"] += len(raw)

        # ── Filtreler (ham sinyal üzerinde) ──
        m_amp  = np.max(np.abs(raw), axis=1) <= AMP_MAX
        m_flat = raw.std(axis=1) >= MIN_STD
        centered = raw - raw.mean(axis=1, keepdims=True)
        zcr      = (np.diff(np.sign(centered), axis=1) != 0).sum(axis=1) / SIGNAL_LENGTH
        m_zcr    = zcr <= MAX_ZCR

        stats["amp_elendi"]  += int((~m_amp).sum())
        stats["flat_elendi"] += int((~m_flat).sum())
        stats["zcr_elendi"]  += int((~m_zcr).sum())

        combined = m_amp & m_flat & m_zcr
        stats["filtre_gecen"] += int(combined.sum())

        if combined.sum() == 0:
            row_cursor = chunk_end
            chunk_idx += 1
            continue

        passed_raw    = raw[combined]
        global_indices= np.where(combined)[0] + s   # global CSV satır numaraları

        # Z-score normalizasyonu
        passed_z = zscore(passed_raw)

        # Pearson korelasyon skoru
        scores = pearson_corr_with_template(passed_z, template)

        # Enerji (IQR filtresi için biriktirilir)
        energies = np.mean(passed_z ** 2, axis=1)

        all_indices.append(global_indices.astype(np.int32))
        all_scores.append(scores)
        all_energies.append(energies)

        row_cursor = chunk_end
        chunk_idx += 1
        if chunk_idx % 5 == 0:
            elapsed = time.time() - t0
            print(f"   Chunk {chunk_idx:3d} | Okunan: {stats['okunan']:6d} | "
                  f"Geçen: {stats['filtre_gecen']:6d} | Süre: {elapsed:.1f}s")

    # Birleştir
    indices  = np.concatenate(all_indices)
    scores   = np.concatenate(all_scores)
    energies = np.concatenate(all_energies)

    # ── Enerji IQR filtresi (global) ──
    q1, q3 = np.percentile(energies, [25, 75])
    iqr     = q3 - q1
    e_mask  = (energies >= q1 - IQR_MULT * iqr) & \
              (energies <= q3 + IQR_MULT * iqr)
    iqr_elim = int((~e_mask).sum())
    indices  = indices[e_mask]
    scores   = scores[e_mask]
    stats["iqr_elendi"] = iqr_elim
    stats["final_filtre_gecen"] = len(indices)

    elapsed_total = time.time() - t0
    print(f"\n   1. Geçiş tamamlandı ({elapsed_total:.1f}s)")
    print(f"   Okunan: {stats['okunan']:,} | Filtreden geçen: {len(indices):,} "
          f"({len(indices)/max(stats['okunan'],1)*100:.1f}%)")

    return indices, scores, stats


# ─────────────────────────────────────────────────────────
# ADIM 3 — Top-K Seç
# ─────────────────────────────────────────────────────────
def select_top_k(indices: np.ndarray,
                 scores:  np.ndarray,
                 k:       int) -> tuple:
    """
    Pearson skoruna göre en yüksek K sinyali seç.
    Döndürür: (seçilen_indices, seçilen_scores)
    """
    k = min(k, len(indices))
    top_order   = np.argsort(scores)[::-1][:k]   # büyükten küçüğe
    selected_ix = indices[top_order]
    selected_sc = scores[top_order]
    print(f"\n[3/4] Top-{k} pseudo-normal sinyal seçildi.")
    print(f"   Skor aralığı: [{selected_sc.min():.4f}, {selected_sc.max():.4f}]")
    print(f"   Ort. korelasyon: {selected_sc.mean():.4f}")
    return np.sort(selected_ix), selected_sc   # indeksleri sıralı tut


# ─────────────────────────────────────────────────────────
# ADIM 4 — Seçilen Satırları Çek (2. Geçiş)
# ─────────────────────────────────────────────────────────
def pass2_extract_signals(target_indices: np.ndarray) -> np.ndarray:
    """
    CSV'yi ikinci kez okur; yalnızca target_indices'teki satırları alır.
    RAM'de yalnızca TOP_K × 4000 × 4 byte (~56 MB) tutar.
    """
    print(f"\n[4/4] Seçilen {len(target_indices):,} satır çekiliyor (2. geçiş)...")
    t0         = time.time()
    idx_set    = set(target_indices.tolist())
    row_cursor = 0
    collected  = []

    for chunk_df in pd.read_csv(TRAIN_CSV, header=None,
                                 chunksize=CHUNK_SIZE, low_memory=True):
        chunk_end = row_cursor + len(chunk_df)

        # Bu chunk'ta hedeflenen satır var mı?
        chunk_start = row_cursor
        in_chunk = [i for i in range(chunk_start, chunk_end) if i in idx_set]
        if in_chunk:
            local_rows = [i - chunk_start for i in in_chunk]
            raw = _fix_cols(chunk_df.iloc[local_rows])
            collected.append(zscore(raw))

        row_cursor = chunk_end
        if row_cursor > int(target_indices.max()) + CHUNK_SIZE:
            break   # hedef satırları geçtik, erken çık

    elapsed = time.time() - t0
    result  = np.vstack(collected)
    print(f"   Çekildi: {result.shape}  ({elapsed:.1f}s)")
    return result


# ─────────────────────────────────────────────────────────
# ADIM 5 — Birleştir & Kaydet
# ─────────────────────────────────────────────────────────
def build_and_save_xtrain(normal_z:    np.ndarray,
                           pseudo_z:   np.ndarray,
                           scores:     np.ndarray,
                           stats:      dict) -> np.ndarray:
    """
    500 etiketli Normal + TOP_K pseudo-Normal → X_train
    Şekil: (N, 4000, 1)  — model girişi için hazır
    """
    combined = np.vstack([normal_z, pseudo_z])          # (500+K, 4000)
    # Karıştır
    perm     = np.random.permutation(len(combined))
    combined = combined[perm]

    X_train  = combined[:, :, np.newaxis].astype(np.float32)  # (N, 4000, 1)
    np.save(XTRAIN_NPY, X_train)

    print(f"\n   X_train kaydedildi: {XTRAIN_NPY}")
    print(f"   Boyut: {X_train.shape}  |  "
          f"Dosya: {os.path.getsize(XTRAIN_NPY)/1e6:.1f} MB")
    print(f"   Mean={X_train.mean():.4f}  Std={X_train.std():.4f}")

    # Rapor
    _write_report(stats, len(normal_z), len(pseudo_z), scores, X_train)
    return X_train


# ─────────────────────────────────────────────────────────
# RAPOR & GÖRSELLEŞTİRME
# ─────────────────────────────────────────────────────────
DARK_BG = "#0d1117"; CARD_BG = "#161b22"
A1 = "#58a6ff"; A2 = "#ff7b72"; A3 = "#3fb950"; TX = "#e6edf3"

plt.rcParams.update({
    "figure.facecolor": DARK_BG, "axes.facecolor": CARD_BG,
    "axes.edgecolor": "#30363d", "axes.labelcolor": TX,
    "xtick.color": TX, "ytick.color": TX, "text.color": TX,
    "grid.color": "#30363d", "grid.alpha": 0.5, "font.size": 9,
})


def _write_report(stats, n_labeled, n_pseudo, scores, X_train):
    sep = "="*60
    lines = [
        sep,
        "  PSEUDO-LABELING PIPELINE RAPORU",
        sep,
        f"  Etiketsiz okunan     : {stats['okunan']:>8,}",
        f"  Genlik filtresi      : -{stats['amp_elendi']:>7,}",
        f"  Düz-çizgi filtresi   : -{stats['flat_elendi']:>7,}",
        f"  ZCR filtresi         : -{stats['zcr_elendi']:>7,}",
        f"  IQR enerji filtresi  : -{stats.get('iqr_elendi',0):>7,}",
        f"  Filtreden geçen      : {stats['final_filtre_gecen']:>8,}",
        "-"*60,
        f"  TOP-K seçilen        : {n_pseudo:>8,}",
        f"  Etiketli Normal      : {n_labeled:>8,}",
        f"  X_train TOPLAM       : {n_labeled+n_pseudo:>8,}",
        f"  Ort. Pearson korel.  : {scores.mean():>8.4f}",
        f"  Min korel. (seçilen) : {scores.min():>8.4f}",
        sep,
        f"  X_train shape        : {X_train.shape}",
        f"  X_train mean/std     : {X_train.mean():.4f} / {X_train.std():.4f}",
        sep,
    ]
    for ln in lines:
        print(ln)
    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n  Rapor: {REPORT_TXT}")

    # ── Görsel ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=DARK_BG)
    fig.suptitle("Pseudo-labeling Pipeline — Özet", fontsize=13,
                 fontweight="bold", color=TX)

    # (a) Korelasyon skoru dağılımı
    ax = axes[0]
    ax.hist(scores, bins=60, color=A1, alpha=0.8, density=True)
    ax.axvline(scores.min(), color=A2, lw=1.5, ls="--",
               label=f"Min seçilen={scores.min():.3f}")
    ax.set_title("Seçilen Pseudo-Normal Korelasyon Dağılımı")
    ax.set_xlabel("Pearson r"); ax.set_ylabel("Yoğunluk")
    ax.legend(fontsize=8); ax.grid(True)

    # (b) Pasta: etiket dağılımı
    ax = axes[1]
    ax.pie([n_labeled, n_pseudo],
           labels=["Etiketli Normal", "Pseudo-Normal"],
           colors=[A3, A1],
           autopct="%1.1f%%",
           startangle=90,
           textprops={"color": TX, "fontsize": 9})
    ax.set_title("X_train Bileşimi")
    ax.set_facecolor(CARD_BG)

    # (c) 3 örnek pseudo-normal sinyal
    ax = axes[2]
    t = np.arange(SIGNAL_LENGTH) / 400.0
    sample_idx = np.random.choice(len(X_train), 3, replace=False)
    for i, idx in enumerate(sample_idx):
        ax.plot(t, X_train[idx, :, 0],
                color=[A1, A3, "#d2a8ff"][i], alpha=0.8,
                lw=0.7, label=f"#{idx}")
    ax.set_title("Örnek X_train Sinyalleri (Z-score)")
    ax.set_xlabel("Zaman (s)"); ax.set_ylabel("Amplitude")
    ax.legend(fontsize=7); ax.grid(True)

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    print(f"  Görsel: {PLOT_PATH}")
    plt.close()


# ─────────────────────────────────────────────────────────
# ANA AKIŞ
# ─────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  ECG Pseudo-labeling Pipeline")
    print("="*60)
    t_start = time.time()

    # 1. Normal şablon
    normal_z  = load_labeled_normal()
    template  = build_template(normal_z)

    # 2. Etiketsiz tara
    indices, scores, stats = pass1_score_unlabeled(template)

    # 3. Top-K seç
    top_indices, top_scores = select_top_k(indices, scores, TOP_K)

    # 4. İkinci geçiş
    pseudo_z = pass2_extract_signals(top_indices)

    # 5. Birleştir & kaydet
    X_train = build_and_save_xtrain(normal_z, pseudo_z, top_scores, stats)

    total_time = time.time() - t_start
    print(f"\n  TAMAMLANDI — {total_time:.1f} saniye")
    print(f"  X_train: {X_train.shape}  → {XTRAIN_NPY}")
    print("\nBu dosyayı ecg_afib_autoencoder.py'de şöyle kullanın:")
    print('  X_train = np.load("Outputs/X_train_pseudo.npy")')
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

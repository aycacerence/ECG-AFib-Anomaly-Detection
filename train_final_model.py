"""
Atriyal Fibrilasyon Tespit Sistemi - Model Eğitim Betiği
-------------------------------------------------------------------------------
Bu betik, hiperparametre optimizasyonu (Grid Search) sonucunda belirlenen
optimal konfigürasyon ile 1D-CAE modelinin final eğitimini gerçekleştirir
ve modeli çıkarım (inference) aşaması için diske kaydeder.
-------------------------------------------------------------------------------
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks
from tensorflow.keras.optimizers import Adam

# Dizin ve Dosya Yolları Konfigürasyonu
OUTPUT_DIR  = "Outputs"
XTRAIN_PATH = os.path.join(OUTPUT_DIR, "X_train_pseudo.npy")
SAVE_PATH   = os.path.join(OUTPUT_DIR, "final_model.keras")

# Global Hiperparametreler
SIGNAL_LENGTH = 4000
LATENT_DIM    = 64
LEARNING_RATE = 1e-3
EPOCHS        = 20
BATCH_SIZE    = 32

def build_1d_cae(latent_dim):
    """
    Belirtilen gizli uzay (latent space) boyutunda, 5 katmanlı 
    Bir Boyutlu Evrişimli Otokodlayıcı (1D-CAE) mimarisini inşa eder.
    """
    inp = layers.Input(shape=(SIGNAL_LENGTH, 1), name="ecg_input")

    # Encoder Ağı
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

    # Decoder Ağı
    enc_len = SIGNAL_LENGTH // 32 # 125
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
    
    out = layers.Conv1DTranspose(1, 7, 2, "same", activation="linear")(x)

    return Model(inp, out, name="Final_1D_CAE")

def main():
    print("="*50)
    print("SİSTEM: Model Eğitim Süreci Başlatılıyor...")
    print("="*50)

    # Veri Yükleme Fazı
    if not os.path.exists(XTRAIN_PATH):
        raise FileNotFoundError(f"Kritik Hata: Eğitim veri seti bulunamadı -> {XTRAIN_PATH}")

    print(f"INFO: Eğitim verisi belleğe alınıyor: {XTRAIN_PATH}")
    X_train = np.load(XTRAIN_PATH)
    print(f"INFO: Veri matrisi boyutu doğrulandı: {X_train.shape}")

    # Model İnisiyalizasyonu ve Derleme
    model = build_1d_cae(LATENT_DIM)
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss="mse")
    
    # Eğitim Konfigürasyonu ve Erken Durdurma (Early Stopping)
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    print(f"\nINFO: Model eğitimi başlatılıyor. (Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE})")
    model.fit(
        X_train, X_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )

    # Seri Hale Getirme (Serialization) ve Diske Yazma
    print(f"\nINFO: Eğitilmiş model ağırlıkları dışa aktarılıyor: {SAVE_PATH}")
    model.save(SAVE_PATH)
    
    print("="*50)
    print("SİSTEM: Model başarıyla eğitildi ve sisteme kaydedildi.")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()

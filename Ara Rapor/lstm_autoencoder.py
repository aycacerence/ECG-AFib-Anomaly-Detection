"""
Atriyal Fibrilasyon Tespit Sistemi - Alternatif Model (Ara Rapor / V1)
-------------------------------------------------------------------------------
Bu modül, zaman serisi analizi için alternatif bir derin öğrenme mimarisi olan
Çift Yönlü Uzun Kısa Vadeli Bellek Otokodlayıcısını (Bidirectional LSTM-AE)
tanımlar.
-------------------------------------------------------------------------------
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

SIGNAL_LENGTH = 4000
LATENT_DIM    = 64
OUTPUT_DIR    = "Outputs"


def build_lstm_autoencoder(signal_length: int = SIGNAL_LENGTH,
                            latent_dim: int = LATENT_DIM) -> Model:
    """
    LSTM-Autoencoder Mimarisi:

    ENCODER
    -------
    Input            (4000, 1)
    Conv1D(8, s=8)   (500,  8)    ← hız için boyut küçültme
    BiLSTM(128)      (500,  256)
    BiLSTM(64)       (500,  128)
    LSTM(latent)     (latent_dim,) ← Latent Vector

    DECODER
    -------
    RepeatVector(500)      (500, latent_dim)
    LSTM(64,  ret_seq=T)   (500, 64)
    LSTM(128, ret_seq=T)   (500, 128)
    TimeDistributed(Dense(8))  (500, 8)
    Flatten → Dense(4000) → Reshape(4000, 1)
    """
    inputs = layers.Input(shape=(signal_length, 1), name="ecg_input")

    # Boyut küçültme (LSTM hızlandırma)
    x = layers.Conv1D(8, kernel_size=8, strides=8, padding="same",
                      activation="relu", name="downsample")(inputs)     # (500, 8)

    # ENCODER
    x = layers.Bidirectional(
            layers.LSTM(128, return_sequences=True, name="enc_lstm1"),
            name="enc_bilstm1")(x)
    x = layers.Dropout(0.2, name="enc_drop1")(x)

    x = layers.Bidirectional(
            layers.LSTM(64, return_sequences=True, name="enc_lstm2"),
            name="enc_bilstm2")(x)
    x = layers.Dropout(0.2, name="enc_drop2")(x)

    latent = layers.LSTM(latent_dim, return_sequences=False,
                          name="bottleneck")(x)                         # (latent_dim,)

    # DECODER
    x = layers.RepeatVector(500, name="repeat")(latent)                 # (500, latent_dim)
    x = layers.LSTM(64,  return_sequences=True, name="dec_lstm1")(x)
    x = layers.Dropout(0.2, name="dec_drop1")(x)
    x = layers.LSTM(128, return_sequences=True, name="dec_lstm2")(x)
    x = layers.Dropout(0.2, name="dec_drop2")(x)
    x = layers.TimeDistributed(layers.Dense(8), name="dec_td")(x)       # (500, 8)
    x = layers.Flatten(name="dec_flatten")(x)                           # (4000,)
    x = layers.Dense(signal_length, activation="linear",
                      name="dec_dense")(x)
    outputs = layers.Reshape((signal_length, 1), name="dec_output")(x)

    model = Model(inputs, outputs, name="LSTM_Autoencoder")
    return model


if __name__ == "__main__":
    print(f"INFO: TensorFlow Versiyonu -> {tf.__version__}")
    print("INFO: LSTM-Autoencoder mimarisi inisiyalize ediliyor...")
    model = build_lstm_autoencoder()
    model.summary()
    print(f"INFO: Toplam Parametre Sayısı: {model.count_params():,}")
    print("INFO: Başarıyla derlendi.")

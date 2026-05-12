"""
Atriyal Fibrilasyon Tespit Sistemi - Çevresel Doğrulama Betiği
-------------------------------------------------------------------------------
Model bağımlılıklarının ve donanım bileşenlerinin (GPU/CPU) test edilerek
çalışma ortamının doğrulandığı analiz modülüdür.
-------------------------------------------------------------------------------
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

print("=" * 55)
print("  Ortam Kontrolu")
print("=" * 55)
print(f"  Python     : {__import__('sys').version.split()[0]}")
print(f"  TensorFlow : {tf.__version__}")
gpus = tf.config.list_physical_devices("GPU")
print(f"  GPU        : {gpus if gpus else 'Yok - CPU modu'}")

print("\n  Gerekli paketler kontrol ediliyor...")
for pkg in ["sklearn", "matplotlib", "pandas"]:
    try:
        __import__(pkg)
        print(f"    OK  {pkg}")
    except ImportError:
        print(f"    EKSIK  {pkg}  <-- pip install {pkg}")

# Model testi
print("\n  1D-CAE model testi (sentetik, 10 ornek)...")
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ecg_afib_autoencoder import build_1d_cae

N       = 10
x_dummy = np.random.randn(N, 4000, 1).astype(np.float32)
model   = build_1d_cae(4000, 64)
model.compile(optimizer="adam", loss="mse")

out = model.predict(x_dummy, verbose=0)
mse = np.mean((x_dummy - out) ** 2)
print(f"  Girdi boyutu     : {x_dummy.shape}")
print(f"  Cikti boyutu     : {out.shape}")
print(f"  Ornek MSE        : {mse:.4f}")
print(f"  Toplam parametre : {model.count_params():,}")

assert out.shape == x_dummy.shape, "HATA: Boyut uyuşmazlığı tespit edildi."
print("\nINFO: Ortam doğrulama süreci başarıyla tamamlandı.")

"""
Atriyal Fibrilasyon Tespit Sistemi - Final Test Arayüzü
-------------------------------------------------------------------------------
Bu modül, eğitilmiş 1D-CAE modelinin çıkarım (inference) performansını
gerçek zamanlı olarak test etmek ve görselleştirmek amacıyla geliştirilmiştir.
-------------------------------------------------------------------------------
"""
import streamlit as st
import numpy as np
import pandas as pd
import time
import plotly.graph_objects as go
import os
import sqlite3
from datetime import datetime
from tensorflow.keras.models import load_model

# Modül Konfigürasyonu ve Arayüz Ayarları
st.set_page_config(
    page_title="Klinik ECG Otonom Analiz",
    layout="wide",
    initial_sidebar_state="expanded"
)

custom_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Roboto', sans-serif !important;
}

/* Siyah/Koyu Lacivert Medikal Tema */
.stApp {
    background-color: #03070c;
}

[data-testid="stSidebar"] {
    background-color: #060b13;
    border-right: 1px solid #1a2332;
}

/* İnce ve Keskin Metrik Kutuları */
[data-testid="stMetric"] {
    background-color: #060b13;
    border: 1px solid #1a2332;
    padding: 10px 15px;
    border-radius: 2px;
    box-shadow: none;
}

[data-testid="stMetricLabel"] {
    font-size: 0.80em;
    color: #8b949e;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

[data-testid="stMetricValue"] {
    font-size: 1.4em;
    color: #c9d1d9;
    font-weight: 300;
}

/* Karar Kutusu (Minimalist Border) */
.decision-box {
    padding: 15px 20px;
    background-color: #060b13;
    border: 1px solid #1a2332;
    border-radius: 2px;
    margin-top: 15px;
}
.decision-normal {
    border-left: 3px solid #3fb950;
}
.decision-afib {
    border-left: 3px solid #ff7b72;
}
.decision-title {
    font-size: 0.95em;
    font-weight: 500;
    margin-bottom: 8px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.decision-normal .decision-title { color: #3fb950; }
.decision-afib .decision-title { color: #ff7b72; }
.decision-text {
    color: #c9d1d9;
    font-size: 0.90em;
    font-weight: 300;
    line-height: 1.5;
}

/* Keskin, Ciddi Buton Tasarımı */
.stButton > button {
    border-radius: 2px;
    border: 1px solid #2a3441;
    background-color: #0a111e;
    color: #c9d1d9;
    font-weight: 400;
    letter-spacing: 0.5px;
    font-size: 0.85em;
    padding: 0.4rem 1rem;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    border-color: #58a6ff;
    color: #58a6ff;
    box-shadow: none;
    background-color: #0d1524;
}

/* Selectbox Minimalizmi */
div[data-baseweb="select"] > div {
    background-color: #0a111e;
    border-color: #2a3441;
    border-radius: 2px;
    color: #c9d1d9;
    font-weight: 300;
}

/* Tab Tasarımı */
[data-baseweb="tab-list"] {
    gap: 10px;
    justify-content: center;
}
[data-baseweb="tab"] {
    background-color: #0a111e !important;
    border-radius: 2px 2px 0 0 !important;
    color: #8b949e !important;
    border: 1px solid #1a2332 !important;
    border-bottom: none !important;
    padding: 10px 20px !important;
}
[aria-selected="true"] {
    background-color: #060b13 !important;
    color: #c9d1d9 !important;
    border-top: 2px solid #58a6ff !important;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Sabit Değişkenler ve Model Konfigürasyonu
MODEL_PATH = "Outputs/final_model.keras"
DATA_PATH  = "Dataset/traindata.csv"
THRESHOLD  = 0.092
SIG_LEN    = 4000

@st.cache_resource
def init_db():
    """SQLite veritabanını başlatır ve tabloyu oluşturur."""
    db_path = "Outputs/patient_records.db"
    db_conn = sqlite3.connect(db_path, check_same_thread=False)
    c = db_conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            patient_id INTEGER,
            mse_score REAL,
            confidence REAL,
            diagnosis TEXT
        )
    ''')
    db_conn.commit()
    return db_conn

# Veritabanı bağlantısını oluştur
conn = init_db()

@st.cache_resource
def load_ecg_model():
    """Disk üzerinden önceden eğitilmiş Keras modelini belleğe alır."""
    if not os.path.exists(MODEL_PATH):
        return None
    return load_model(MODEL_PATH)

@st.cache_data
def load_all_signals():
    """
    Hedef veri setinden ilk 1000 satırı belleğe alır ve
    Z-Skoru normalizasyonu uygular.
    """
    if not os.path.exists(DATA_PATH):
        return None
    df = pd.read_csv(DATA_PATH, header=None, nrows=1000)
    data = df.iloc[:, :SIG_LEN].values.astype(np.float32)
    mu = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True) + 1e-8
    return (data - mu) / std

def calculate_confidence(mse, threshold):
    """
    MSE değerinin anomali eşiğine olan asimptotik uzaklığını
    hesaplayarak 50.0 ile 99.9 arasında bir güven skoru üretir.
    """
    diff = abs(mse - threshold) / threshold
    conf = 50 + 50 * (1 - np.exp(-1.2 * diff))
    return min(99.9, conf)

# Ana Akış Denetimi
st.markdown("<h1 style='text-align: center; color: #c9d1d9; font-weight: 300; letter-spacing: 1px; font-size: 1.6em; border-bottom: 1px solid #1a2332; padding-bottom: 15px; margin-bottom: 30px;'>ECG ATRİYAL FİBRİLASYON OTONOM TEŞHİS SİSTEMİ</h1>", unsafe_allow_html=True)

# Sekmeler (Tabs)
tab_monitor, tab_db = st.tabs(["CANLI KLİNİK MONİTÖR", "HASTA VERİTABANI GEÇMİŞİ"])

# Yan Menü Konfigürasyonu
with st.sidebar:
    st.markdown("<h3 style='color: #8b949e; font-weight: 400; font-size: 1.0em; border-bottom: 1px solid #1a2332; padding-bottom: 10px; margin-bottom: 20px; text-transform: uppercase;'>HASTA KAYIT VERİTABANI</h3>", unsafe_allow_html=True)
    
    selected_patient_idx = st.number_input("Hasta Protokol Numarası", min_value=0, max_value=999, value=510, step=1)
    
    st.markdown("""
<div style='color: #888888; font-size: 13px; margin-top: 10px; padding: 10px; border: 1px solid #333; border-radius: 5px;'>
<b>VERİTABANI REFERANSI:</b><br>
[000 - 499] : Patolojik Kayıtlar (AFib)<br>
[500 - 999] : Sağlıklı Kayıtlar (Normal)
</div>
""", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("Hasta Verisini Yükle ve Analiz Et", use_container_width=True)
    
    st.markdown("---")
    st.markdown(f"""
    <div style='color: #8b949e; font-size: 0.85em; margin-bottom: 5px; font-weight: 500;'>SİSTEM KONFİGÜRASYONU:</div>
    <div style='color: #c9d1d9; font-size: 0.80em; line-height: 1.6;'>
    <b>Algoritma:</b> 1D-CAE (Autoencoder)<br>
    <b>Latent Uzayı (Z):</b> 64 Boyutlu<br>
    <b>Parametre Hacmi:</b> ~4.8 Milyon<br>
    <b>Örnekleme Frekansı:</b> 400 Hz<br>
    <b>Sinyal Çözünürlüğü:</b> 4000 Nokta/Kayıt<br>
    <b>Optimal Karar Eşiği:</b> {THRESHOLD}<br>
    <b>Çıkarım (Inference):</b> Gerçek Zamanlı
    </div>
    """, unsafe_allow_html=True)

with tab_db:
    st.markdown("<br>", unsafe_allow_html=True)
    try:
        df_db = pd.read_sql_query("SELECT timestamp as 'Kayıt Tarihi', patient_id as 'Protokol No', ROUND(mse_score, 4) as 'Ölçülen MSE', ROUND(confidence, 1) as 'Güven Skoru (%)', diagnosis as 'Sistem Kararı' FROM analyses ORDER BY id DESC", conn)
        if not df_db.empty:
            st.dataframe(df_db, use_container_width=True, hide_index=True)
        else:
            st.info("Veritabanında henüz hasta kaydı bulunmamaktadır. Lütfen canlı monitör üzerinden analiz başlatınız.")
    except Exception as e:
        st.error(f"Veritabanı okuma hatası: {e}")

with tab_monitor:
    # Bağımlılıkların Yüklenmesi
    model = load_ecg_model()
    signals = load_all_signals()

    if model is None or signals is None:
        st.error("Sistem Hatası: Model veya Veri Seti bağlantısı kurulamadı. Yolları kontrol ediniz.")
        st.stop()

    # Analiz ve Görselleştirme Döngüsü
    if analyze_btn:
        current_idx = selected_patient_idx
        signal = signals[current_idx]

        # Modele veri beslenmesi ve tahmin (Inference)
        input_sig = signal[np.newaxis, :, np.newaxis]
        recon_sig = model.predict(input_sig, verbose=0)[0, :, 0]
        mse = float(np.mean((signal - recon_sig)**2))

        status = "NORMAL" if mse < THRESHOLD else "AFIB"
        conf = calculate_confidence(mse, THRESHOLD)

        # SQLite Veritabanına Kayıt İşlemi
        c = conn.cursor()
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute("INSERT INTO analyses (timestamp, patient_id, mse_score, confidence, diagnosis) VALUES (?, ?, ?, ?, ?)",
                  (current_time, current_idx, mse, conf, status))
        conn.commit()

        st.markdown(f"<h3 style='color: #8b949e; font-size: 1.1em; font-weight: 400;'>AKTİF MONİTÖR: <span style='color: #c9d1d9;'>Protokol #{current_idx:03d}</span></h3>", unsafe_allow_html=True)

        # UI Placeholder
        chart_placeholder = st.empty()

        # Çizim Parametreleri
        chunk_size = 80
        frames = SIG_LEN // chunk_size

        # Figür Optimizasyonu: Yalnızca bir kez başlatılır
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(y=[], mode='lines', name='ORİJİNAL EKG', 
                                 line=dict(color='#00f2fe', width=1.5)))
        
        fig.add_trace(go.Scatter(y=[], mode='lines', name='SİSTEM REKONSTRÜKSİYONU', 
                                 line=dict(color='#fe0979', width=1.5, dash='dot')))
        
        # Layout Ayarları
        # Y-ekseni Z-skoru standartlarına göre sabitlenmiştir [-5, 5]
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(range=[0, SIG_LEN], showgrid=False, zeroline=False, visible=False),
            yaxis=dict(range=[-5, 5], showgrid=False, zeroline=False, color='#484f58', tickfont=dict(size=10, family="monospace")),
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5, font=dict(size=10, color="#8b949e", family="Roboto")),
            height=350
        )

        # Zaman serisi verisinin ardışık (sequential) olarak güncellenmesi
        for i in range(1, frames + 1):
            end_idx = i * chunk_size if i < frames else SIG_LEN
            
            y_orig = np.full(SIG_LEN, np.nan)
            y_orig[:end_idx] = signal[:end_idx]
            
            y_recon = np.full(SIG_LEN, np.nan)
            y_recon[:end_idx] = recon_sig[:end_idx]
            
            # In-place veri mutasyonu
            fig.data[0].y = y_orig
            fig.data[1].y = y_recon
            
            # Arayüz güncellenmesi
            chart_placeholder.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            time.sleep(0.04)

        # Analiz Çıktıları ve Karar Mekanizması
        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            st.metric(label="REKONSTRÜKSİYON HATASI (MSE)", value=f"{mse:.5f}", delta=f"{mse - THRESHOLD:.5f} Sınır Farkı", delta_color="inverse")

        with col2:
            st.metric(label="KARAR EŞİĞİ (THRESHOLD)", value=f"{THRESHOLD:.3f}")

        with col3:
            if status == "NORMAL":
                html = f"""
                <div class="decision-box decision-normal">
                    <div class="decision-title">SİSTEM KARARI: NORMAL SİNÜS RİTMİ</div>
                    <div class="decision-text">
                        Hesaplanan Güven Skoru: <b>%{conf:.1f}</b><br>
                        Klinik Ön Değerlendirme: Hastanın EKG trasesinde rekonstrüksiyon sınırlarını aşan anomali (atriyal fibrilasyon bulgusu) saptanmamıştır. Ritim stabildir.
                    </div>
                </div>
                """
            else:
                html = f"""
                <div class="decision-box decision-afib">
                    <div class="decision-title">SİSTEM KARARI: ATRİYAL FİBRİLASYON (AFib)</div>
                    <div class="decision-text">
                        Hesaplanan Güven Skoru: <b>%{conf:.1f}</b><br>
                        Klinik Ön Değerlendirme: Hastanın EKG trasesinde rekonstrüksiyon eşiğini aşan, yüksek varyanslı ritim bozukluğu saptanmıştır. Uzman kontrolü önerilir.
                    </div>
                </div>
                """
            st.markdown(html, unsafe_allow_html=True)

    else:
        # Bekleme Durumu İşleyicisi
        st.markdown(
            "<div style='text-align: center; color: #484f58; margin-top: 100px; font-weight: 300; font-size: 1.1em;'>"
            "Lütfen yan paneldeki veritabanından bir hasta kaydı seçin ve analiz sürecini başlatın."
            "</div>", 
            unsafe_allow_html=True
        )

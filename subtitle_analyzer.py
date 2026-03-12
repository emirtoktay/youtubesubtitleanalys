import re
import json
import gc  # RAM temizliği için
import numpy as np
import joblib
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

# Model 1: LSTM (TensorFlow/Keras)
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder as LSTM_LabelEncoder

# Model 2: BERT (Hugging Face Transformers/PyTorch)
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# GPU aramasın, direkt CPU kullansın
tf.config.set_visible_devices([], 'GPU')


# ==============================
# 🔹 MODEL YÜKLEME FONKSİYONLARI
# ==============================
def load_lstm_model():
    try:
        model = load_model("turkish_toxic_lstm_model_full.h5")
        with open("label_encoder.json", "r", encoding="utf-8") as f:
            le_data = json.load(f)
        le = LSTM_LabelEncoder()
        le.classes_ = np.array(le_data["classes"])
        with open("tokenizer.json", "r", encoding="utf-8") as f:
            tokenizer = tokenizer_from_json(f.read())
        return model, tokenizer, le
    except Exception as e:
        print(f"❌ LSTM yükleme hatası: {e}")
        return None, None, None


def load_bert_model():
    try:
        MODEL_DIR = "armud/emir-toxic-bert"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        model.eval()
        device = torch.device("cpu")
        model.to(device)
        with open("label_encoder.json", "r", encoding="utf-8") as f:
            le_data = json.load(f)
        le = LSTM_LabelEncoder()
        le.classes_ = np.array(le_data["classes"])
        return model, tokenizer, le, device
    except Exception as e:
        print(f"❌ BERT yükleme hatası: {e}")
        return None, None, None, None


def load_svc_model():
    try:
        model = joblib.load("linear_svc_model.pkl")
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        return model, vectorizer
    except Exception as e:
        print(f"❌ SVC yükleme hatası: {e}")
        return None, None


# ===================================================
# 🔹 ALTYAZI ÇEKME
# ===================================================
# ===================================================
# 🔹 ALTYAZI ÇEKME
# ===================================================
# ===================================================
# 🔹 ALTYAZI ÇEKME (GÜNCELLENDİ)
# ===================================================
def get_caption_with_yta(video_id: str):
    print(f"🔍 Altyazı aranıyor... Video ID: {video_id}")
    try:
        # 💡 YENİ YÖNTEM: Hata veren get_transcript yerine list_transcripts kullanıyoruz
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        # Direkt olarak 'tr' (Türkçe) altyazıyı bul ve çek
        transcript = transcript_list.find_transcript(['tr'])
        lines = transcript.fetch()

        print(f"✅ Altyazı başarıyla çekildi: {len(lines)} satır.")

    except NoTranscriptFound:
        print("⚠️ DİKKAT: Bu videoda Türkçe (tr) altyazı bulunamadı!")
        return []
    except TranscriptsDisabled:
        print("⚠️ DİKKAT: Bu video için altyazılar tamamen kapatılmış!")
        return []
    except Exception as e:
        print(f"⚠️ DİKKAT: Altyazı çekerken bilinmeyen bir sorun oldu: {e}")
        return []

    captions = []
    for line in lines:
        text = line['text'].strip()
        if not text or re.fullmatch(r"[\[\(].*[\]\)]", text.strip()): continue
        # Küfür filtrelemesi
        text = text.replace("[__]", "siktir").replace("[ __ ]", "amk").replace("[\xa0__\xa0]", "amk")
        captions.append({
            "text": text,
            "start": round(line['start'], 2),
            "end": round(line['start'] + line['duration'], 2)
        })
    return captions

# ===================================================
# 🔹 TAHMİN FONKSİYONLARI (Artık modelleri parametre olarak alıyor)
# ===================================================
def predict_text_lstm(text, model, tokenizer, le):
    if model is None: return "MODEL_HATA"
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=100, padding='post', truncating='post')
    preds = model.predict(padded, verbose=0)
    label_index = np.argmax(preds)
    return le.inverse_transform([label_index])[0]


def predict_text_bert(text, model, tokenizer, le, device):
    if model is None: return "MODEL_HATA"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    label_index = np.argmax(probs)
    return le.inverse_transform([label_index])[0]


def predict_text_svc(text, model, vectorizer):
    if model is None: return "MODEL_HATA"
    vec = vectorizer.transform([text])
    return model.predict(vec)[0]


# ===================================================
# 🔹 ANA ANALİZ FONKSİYONU (SİHİR BURADA GERÇEKLEŞİYOR)
# ===================================================
# ===================================================
# 🔹 ANA ANALİZ FONKSİYONU (SIRALI YÜKLEME VE SİLME)
# ===================================================
def analyze_subtitles(video_id):
    captions = get_caption_with_yta(video_id)
    if not captions:
        return None

    total_lines = len(captions)
    safe_counts = {"lstm": 0, "bert": 0, "svc": 0}

    print(f"🚀 {total_lines} satır altyazı bulundu. Sıralı analiz başlıyor...")

    # ---------------------------------------------------
    # 1. AŞAMA: SADECE LSTM
    # ---------------------------------------------------
    print("⏳ 1/3: LSTM Modeli RAM'e yükleniyor...")
    lstm_m, lstm_t, lstm_le = load_lstm_model()
    if lstm_m is not None:
        for c in captions:
            l_label = predict_text_lstm(c['text'], lstm_m, lstm_t, lstm_le)
            if l_label == "OTHER":
                safe_counts["lstm"] += 1
    print("🧹 LSTM işi bitti, RAM'den siliniyor...")
    del lstm_m, lstm_t, lstm_le
    gc.collect()

    # ---------------------------------------------------
    # 2. AŞAMA: SADECE BERT (En Ağırı)
    # ---------------------------------------------------
    print("⏳ 2/3: BERT Modeli RAM'e yükleniyor...")
    bert_m, bert_t, bert_le, bert_d = load_bert_model()
    if bert_m is not None:
        for c in captions:
            b_label = predict_text_bert(c['text'], bert_m, bert_t, bert_le, bert_d)
            if b_label == "OTHER":
                safe_counts["bert"] += 1
    print("🧹 BERT işi bitti, RAM'den siliniyor...")
    del bert_m, bert_t, bert_le, bert_d
    gc.collect()

    # ---------------------------------------------------
    # 3. AŞAMA: SADECE SVC
    # ---------------------------------------------------
    print("⏳ 3/3: Linear SVC Modeli RAM'e yükleniyor...")
    svc_m, svc_v = load_svc_model()
    if svc_m is not None:
        for c in captions:
            s_label = predict_text_svc(c['text'], svc_m, svc_v)
            if s_label == "OTHER":
                safe_counts["svc"] += 1
    print("🧹 SVC işi bitti, RAM'den siliniyor...")
    del svc_m, svc_v
    gc.collect()

    print("✨ Tüm analizler bitti, RAM tertemiz!")

    return {
        "percentages": {
            "lstm": round((safe_counts["lstm"] / total_lines) * 100, 2) if total_lines > 0 else 100.0,
            "bert": round((safe_counts["bert"] / total_lines) * 100, 2) if total_lines > 0 else 100.0,
            "svc": round((safe_counts["svc"] / total_lines) * 100, 2) if total_lines > 0 else 100.0,
        },
        "total_lines": total_lines
    }
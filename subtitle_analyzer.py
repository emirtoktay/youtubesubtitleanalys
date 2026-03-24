import re
import json
import gc
import requests
import numpy as np
import joblib

# SADECE PLAN B: YT-DLP
import yt_dlp

# Model 1: LSTM
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder as LSTM_LabelEncoder

# Model 2: BERT
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

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
# 🔹 ALTYAZI ÇEKME (PLAN B: YT-DLP ANDROID TAKLİDİ)
# ===================================================
def get_ytdlp_captions(video_id):
    url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = {
        'skip_download': True, 'writesubtitles': True, 'writeautomaticsub': True,
        'subtitleslangs': ['tr'], 'subtitlesformat': 'json3',
        'extractor_args': {
            'youtube': {
                'player_client': ['android', 'ios', 'mweb'],
                'player_skip': ['webpage', 'configs']
            }
        },
        'user_agent': 'Mozilla/5.0 (Linux; Android 14; SM-S918B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Mobile Safari/537.36',
        'quiet': True, 'no_warnings': True, 'nocheckcertificate': True
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            subs = info.get('requested_subtitles', {})
            if not subs or 'tr' not in subs: return []
            sub_url = subs['tr'].get('url')
            if not sub_url: return []

            resp = requests.get(sub_url, timeout=10)
            data = resp.json()
            captions = []
            for event in data.get('events', []):
                if 'segs' in event:
                    text = "".join([seg.get('utf8', '') for seg in event['segs']]).strip()
                    if not text or re.fullmatch(r"[\[\(].*[\]\)]", text.strip()): continue
                    text = text.replace("[__]", "siktir").replace("[ __ ]", "amk").replace("[\xa0__\xa0]", "amk")
                    start = event.get('tStartMs', 0) / 1000.0
                    duration = event.get('dDurationMs', 0) / 1000.0
                    captions.append({"text": text, "start": round(start, 2), "end": round(start + duration, 2)})
            return captions
    except Exception as e:
        print(f"⚠️ YT-DLP Android taklidi başarısız oldu: {e}")
        return []


# ===================================================
# 🔹 TAHMİN FONKSİYONLARI
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
# 🔹 ANA ANALİZ FONKSİYONU
# ===================================================
def analyze_subtitles(video_id):
    print(f"🔍 [PLAN B] yt-dlp ile altyazı çekiliyor... ({video_id})")
    captions = get_ytdlp_captions(video_id)

    if not captions:
        print("🛑 Altyazı çekilemedi.")
        return None

    total_lines = len(captions)
    safe_counts = {"lstm": 0, "bert": 0, "svc": 0}

    print(f"🚀 {total_lines} satır altyazı bulundu. Sıralı analiz başlıyor...")

    print("⏳ 1/3: LSTM Modeli RAM'e yükleniyor...")
    lstm_m, lstm_t, lstm_le = load_lstm_model()
    if lstm_m is not None:
        for c in captions:
            if predict_text_lstm(c['text'], lstm_m, lstm_t, lstm_le) == "OTHER":
                safe_counts["lstm"] += 1
    del lstm_m, lstm_t, lstm_le
    gc.collect()

    print("⏳ 2/3: BERT Modeli RAM'e yükleniyor...")
    bert_m, bert_t, bert_le, bert_d = load_bert_model()
    if bert_m is not None:
        for c in captions:
            if predict_text_bert(c['text'], bert_m, bert_t, bert_le, bert_d) == "OTHER":
                safe_counts["bert"] += 1
    del bert_m, bert_t, bert_le, bert_d
    gc.collect()

    print("⏳ 3/3: Linear SVC Modeli RAM'e yükleniyor...")
    svc_m, svc_v = load_svc_model()
    if svc_m is not None:
        for c in captions:
            if predict_text_svc(c['text'], svc_m, svc_v) == "OTHER":
                safe_counts["svc"] += 1
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
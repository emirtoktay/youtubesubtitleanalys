import re
import json
import gc
import os
import numpy as np
import joblib
import concurrent.futures  # YENİ: Gölge ban koruması ve zaman aşımı için

# YENİ SİLAHIMIZ
from youtube_transcript_api import YouTubeTranscriptApi

# Model 1: LSTM
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder as LSTM_LabelEncoder

# Model 2: BERT
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
# 🔹 ALTYAZI ÇEKME (ZIRHLI & ÇEREZSİZ VERSİYON)
# ===================================================
def fetch_api(video_id):
    """Sadece API'ye istek atan saf fonksiyon (Thread içinde çalışacak)"""
    if os.path.exists('cookies.txt'):
        return YouTubeTranscriptApi.list_transcripts(video_id, cookies='cookies.txt')
    else:
        return YouTubeTranscriptApi.list_transcripts(video_id)


def get_caption_with_yta(video_id: str):
    print(f"🔍 youtube-transcript-api ile altyazı aranıyor... Video ID: {video_id}")

    transcript_list = None

    # 🛡️ GÖLGE BAN (SHADOWBAN) KORUMASI: 10 Saniye Şalteri
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(fetch_api, video_id)
            # 10 saniye bekle, YouTube cevap vermezse işlemi öldür!
            transcript_list = future.result(timeout=10)

    except concurrent.futures.TimeoutError:
        print("🛑 KORUMA DEVREDE: YouTube gölge ban uyguluyor (asılı kalma). Sunucu kilitlenmesi başarıyla önlendi!")
        return []
    except Exception as e:
        print(f"⚠️ Altyazı listesi çekilemedi. Hata: {e}")
        return []

    if not transcript_list:
        return []

    try:
        transcript = None
        # Önce Türkçe ara, yoksa otomatik oluşturulan Türkçe, o da yoksa İngilizceyi çevir
        try:
            transcript = transcript_list.find_transcript(['tr'])
        except:
            try:
                transcript = transcript_list.find_generated_transcript(['tr'])
            except:
                for t in transcript_list:
                    if t.is_translatable:
                        transcript = t.translate('tr')
                        break

        if not transcript:
            print("⚠️ DİKKAT: Türkçe altyazı bulunamadı!")
            return []

        data = transcript.fetch()
        captions = []

        for item in data:
            text = item.get('text', '').strip()

            if not text or re.fullmatch(r"[\[\(].*[\]\)]", text):
                continue

            # Küfür düzeltmeleri
            text = text.replace("[__]", "siktir").replace("[ __ ]", "amk").replace("[\xa0__\xa0]", "amk")
            text = text.replace("\n", " ")

            start = float(item.get('start', 0))
            duration = float(item.get('duration', 0))

            captions.append({
                "text": text,
                "start": round(start, 2),
                "end": round(start + duration, 2)
            })

        print(f"✅ Başarıyla çekildi: {len(captions)} satır.")
        return captions

    except Exception as e:
        print(f"⚠️ Altyazı verisi okunurken hata: {e}")
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
    captions = get_caption_with_yta(video_id)
    if not captions:
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
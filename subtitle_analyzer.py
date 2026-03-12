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

# GPU aramasın, direkt CPU kullansın (Railway için zorunlu)
tf.config.set_visible_devices([], 'GPU')


# ==============================
# 🔹 MODEL ve DOSYALARI YÜKLEME
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
        print("✅ LSTM Modeli yüklendi.")
        return model, tokenizer, le
    except Exception as e:
        print(f"❌ LSTM modeli yüklenirken hata: {e}")
        return None, None, None


def load_bert_model():
    try:
        # 💡 YENİ: Klasörden değil, direkt senin Hugging Face depodan çekiyor!
        MODEL_DIR = "armud/emir-toxic-bert"

        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        model.eval()
        device = torch.device("cpu")
        model.to(device)

        # Label Encoder'ı hala ana klasöründeki json'dan okumaya devam ediyor
        with open("label_encoder.json", "r", encoding="utf-8") as f:
            le_data = json.load(f)
        le = LSTM_LabelEncoder()
        le.classes_ = np.array(le_data["classes"])

        print(f"✅ BERT Modeli Hugging Face'ten yüklendi. (Cihaz: {device})")
        return model, tokenizer, le, device
    except Exception as e:
        print(f"❌ BERT modeli yüklenirken hata: {e}")
        return None, None, None, None


def load_svc_model():
    try:
        model = joblib.load("linear_svc_model.pkl")
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        print("✅ Linear SVC Modeli yüklendi.")
        return model, vectorizer
    except Exception as e:
        print(f"❌ Linear SVC modeli yüklenirken hata: {e}")
        return None, None


# Sunucu başlarken modelleri 1 kere yüklüyoruz
LSTM_MODEL, LSTM_TOKENIZER, LSTM_LE = load_lstm_model()
BERT_MODEL, BERT_TOKENIZER, BERT_LE, BERT_DEVICE = load_bert_model()
SVC_MODEL, SVC_VECTORIZER = load_svc_model()


# ===================================================
# 🔹 ALTYAZI ÇEKME (DÜZELTİLDİ)
# ===================================================
def get_caption_with_yta(video_id: str):
    try:
        # Hata buradaydı, doğru kullanım bu şekilde:
        lines = YouTubeTranscriptApi.get_transcript(video_id, languages=['tr'])
    except NoTranscriptFound:
        print(f"⚠️ Türkçe altyazı bulunamadı ({video_id}).")
        return []
    except TranscriptsDisabled:
        print(f"⚠️ Altyazılar bu video için kapalı ({video_id}).")
        return []
    except Exception as e:
        print(f"⚠️ Altyazı çekerken bilinmeyen hata: {e}")
        return []

    captions = []
    for line in lines:
        text = line['text'].strip()  # 'text' bir dictionary key'idir
        if not text or re.fullmatch(r"[\[\(].*[\]\)]", text.strip()): continue
        text = text.replace("[__]", "siktir").replace("[ __ ]", "amk").replace("[\xa0__\xa0]", "amk")
        captions.append({
            "text": text,
            "start": round(line['start'], 2),
            "end": round(line['start'] + line['duration'], 2)
        })
    return captions


# ===================================================
# 🔹 TAHMİN FONKSİYONLARI
# ===================================================
def predict_text_lstm(text):
    if LSTM_MODEL is None: return "MODEL_HATA", 0.0
    seq = LSTM_TOKENIZER.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=100, padding='post', truncating='post')
    preds = LSTM_MODEL.predict(padded, verbose=0)
    label_index = np.argmax(preds)
    return LSTM_LE.inverse_transform([label_index])[0], float(np.max(preds))


def predict_text_bert(text):
    if BERT_MODEL is None: return "MODEL_HATA", 0.0
    inputs = BERT_TOKENIZER(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(BERT_DEVICE)
    with torch.no_grad():
        outputs = BERT_MODEL(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    label_index = np.argmax(probs)
    return BERT_LE.inverse_transform([label_index])[0], float(np.max(probs))


def predict_text_svc(text):
    if SVC_MODEL is None: return "MODEL_HATA", 0.0
    vec = SVC_VECTORIZER.transform([text])
    label = SVC_MODEL.predict(vec)[0]
    try:
        conf = float(np.max(SVC_MODEL.decision_function(vec)))
    except Exception:
        conf = 0.0
    return label, conf


# ===================================================
# 🔹 ANA ANALİZ FONKSİYONU
# ===================================================
def analyze_subtitles(video_id):
    captions = get_caption_with_yta(video_id)
    if not captions:
        return None

    safe_counts = {"lstm": 0, "bert": 0, "svc": 0}

    for c in captions:
        text = c['text']
        l_label, _ = predict_text_lstm(text)
        b_label, _ = predict_text_bert(text)
        s_label, _ = predict_text_svc(text)

        if l_label == "OTHER": safe_counts["lstm"] += 1
        if b_label == "OTHER": safe_counts["bert"] += 1
        if s_label == "OTHER": safe_counts["svc"] += 1

    # İşlem bittikten sonra RAM'i rahatlat
    gc.collect()

    total_lines = len(captions)
    return {
        "percentages": {
            "lstm": round((safe_counts["lstm"] / total_lines) * 100, 2),
            "bert": round((safe_counts["bert"] / total_lines) * 100, 2),
            "svc": round((safe_counts["svc"] / total_lines) * 100, 2),
        },
        "total_lines": total_lines
    }
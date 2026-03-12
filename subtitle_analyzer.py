import re
import json
import gc  # RAM temizliği için
import numpy as np
import requests  # YENİ
import yt_dlp  # YENİ (Sorunsuz altyazı çekici)


# DİKKAT: tensorflow, torch, transformers, joblib ve sklearn importları
# buradan tamamen silindi. RAM şişmesin diye sadece video gelince yüklenecekler!

# ==============================
# 🔹 MODEL YÜKLEME FONKSİYONLARI (Tembel Yükleme)
# ==============================
def load_lstm_model():
    # Sadece fonksiyon çağrıldığında RAM'e alınır
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.text import tokenizer_from_json
    from sklearn.preprocessing import LabelEncoder as LSTM_LabelEncoder

    # GPU aramasın, direkt CPU kullansın
    tf.config.set_visible_devices([], 'GPU')

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
    # Sadece fonksiyon çağrıldığında RAM'e alınır
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from sklearn.preprocessing import LabelEncoder as LSTM_LabelEncoder
    import torch

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
    # Sadece fonksiyon çağrıldığında RAM'e alınır
    import joblib

    try:
        model = joblib.load("linear_svc_model.pkl")
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        return model, vectorizer
    except Exception as e:
        print(f"❌ SVC yükleme hatası: {e}")
        return None, None


# ===================================================
# 🔹 ALTYAZI ÇEKME (YT-DLP + ANDROID TAKLİDİ!)
# ===================================================
def get_caption_with_yta(video_id: str):
    print(f"🔍 yt-dlp ile (Android Taklidi) altyazı aranıyor... Video ID: {video_id}")
    url = f"https://www.youtube.com/watch?v={video_id}"

    # Senin bulduğun efsanevi çerezsiz atlatma yöntemi + Altyazı ayarları
    ydl_opts = {
        'skip_download': True,  # Sadece altyazı istiyoruz, video indirme
        'writesubtitles': True,
        'writeautomaticsub': True,  # Otomatik TR altyazıları da al
        'subtitleslangs': ['tr'],
        'subtitlesformat': 'json3',

        # 🚀 İŞTE YOUTUBE'U KANDIRAN O SİHİRLİ KISIM (Senin Kodun)
        'extractor_args': {
            'youtube': {
                'player_client': ['android', 'ios']
            }
        },
        'user_agent': 'android-app/com.google.android.youtube/19.05.36 (Linux; U; Android 14; tr_TR)',

        'quiet': True,
        'no_warnings': True
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

            # Altyazı var mı kontrol et
            subs = info.get('requested_subtitles', {})
            if not subs or 'tr' not in subs:
                print("⚠️ DİKKAT: Bu videoda normal veya otomatik Türkçe (tr) altyazı bulunamadı!")
                return []

            # Altyazı dosyasının JSON linkini al
            sub_url = subs['tr'].get('url')
            if not sub_url:
                return []

            # Linkten JSON verisini anında indir ve parçala
            resp = requests.get(sub_url)
            data = resp.json()

            captions = []
            for event in data.get('events', []):
                if 'segs' in event:
                    # Kelimeleri birleştirip tek satır yap
                    text = "".join([seg.get('utf8', '') for seg in event['segs']]).strip()

                    if not text or re.fullmatch(r"[\[\(].*[\]\)]", text.strip()):
                        continue

                    # Küfür filtrelemesi
                    text = text.replace("[__]", "siktir").replace("[ __ ]", "amk").replace("[\xa0__\xa0]", "amk")

                    start = event.get('tStartMs', 0) / 1000.0
                    duration = event.get('dDurationMs', 0) / 1000.0

                    captions.append({
                        "text": text,
                        "start": round(start, 2),
                        "end": round(start + duration, 2)
                    })

            print(f"✅ Altyazı yt-dlp (Android) ile başarıyla çekildi: {len(captions)} satır.")
            return captions

    except Exception as e:
        print(f"⚠️ DİKKAT: yt-dlp altyazı çekerken hata fırlattı: {e}")
        return []


# ===================================================
# 🔹 TAHMİN FONKSİYONLARI
# ===================================================
def predict_text_lstm(text, model, tokenizer, le):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    if model is None: return "MODEL_HATA"
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=100, padding='post', truncating='post')
    preds = model.predict(padded, verbose=0)
    label_index = np.argmax(preds)
    return le.inverse_transform([label_index])[0]


def predict_text_bert(text, model, tokenizer, le, device):
    import torch
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

    # Keras backend'i temizlemek için tf'yi buraya çağırıyoruz
    import tensorflow as tf
    tf.keras.backend.clear_session()
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
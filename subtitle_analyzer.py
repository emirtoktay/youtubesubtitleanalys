import re
import json
import gc
import requests
import numpy as np
import joblib
import concurrent.futures

# ÇİFT MOTORLU + AYNA SUNUCU SİSTEMİ
from youtube_transcript_api import YouTubeTranscriptApi
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
# 🔹 ALTYAZI ÇEKME (PLAN A: YTA API - ÇEREZSİZ)
# ===================================================
def fetch_api(video_id):
    return YouTubeTranscriptApi.list_transcripts(video_id)


def get_yta_captions(video_id):
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(fetch_api, video_id)
            transcript_list = future.result(timeout=10)

        transcript = None
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
        if not transcript: return []

        data = transcript.fetch()
        captions = []
        for item in data:
            text = item.get('text', '').strip()
            if not text or re.fullmatch(r"[\[\(].*[\]\)]", text): continue
            text = text.replace("[__]", "siktir").replace("[ __ ]", "amk").replace("[\xa0__\xa0]", "amk").replace("\n",
                                                                                                                  " ")
            start = float(item.get('start', 0))
            duration = float(item.get('duration', 0))
            captions.append({"text": text, "start": round(start, 2), "end": round(start + duration, 2)})
        return captions
    except Exception as e:
        print(f"⚠️ YTA API başarısız oldu: {e}")
        return []


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
        print(f"⚠️ YT-DLP Android taklidi de başarısız oldu: {e}")
        return []


# ===================================================
# 🔹 ALTYAZI ÇEKME (PLAN C: FARKLI SUNUCULAR - PIPED API)
# ===================================================
def get_plan_c_captions(video_id):
    print(f"🌍 [PLAN C] Alternatif Ayna Sunucular (Piped API) deneniyor... ({video_id})")

    instances = [
        "https://pipedapi.kavin.rocks",
        "https://pipedapi.tokhmi.xyz",
        "https://pipedapi.smnz.de"
    ]

    for base_url in instances:
        try:
            print(f"🔄 İstek atılıyor: {base_url} ...")
            api_url = f"{base_url}/streams/{video_id}"
            resp = requests.get(api_url, timeout=10)
            if resp.status_code != 200:
                continue

            data = resp.json()
            subtitles = data.get('subtitles', [])

            tr_url = None
            for sub in subtitles:
                if sub.get('code') == 'tr' or 'Turkish' in sub.get('name', ''):
                    tr_url = sub.get('url')
                    break

            if not tr_url:
                continue

            sub_resp = requests.get(tr_url, timeout=10)
            vtt_text = sub_resp.text

            captions = []
            blocks = re.split(r'\n\n+', vtt_text)

            for block in blocks:
                lines = block.strip().split('\n')
                time_line = ""
                text_lines = []
                for line in lines:
                    if '-->' in line:
                        time_line = line
                    elif time_line and line.strip() and not line.startswith('WEBVTT'):
                        text_lines.append(line.strip())

                if time_line and text_lines:
                    text = " ".join(text_lines)
                    text = re.sub(r'<[^>]+>', '', text)
                    text = text.replace("[__]", "siktir").replace("[ __ ]", "amk")
                    captions.append({"text": text, "start": 0.0, "end": 0.0})

            if captions:
                return captions
        except Exception as e:
            print(f"⚠️ Sunucu ({base_url}) hata verdi: {e}. Diğerine geçiliyor...")
            continue

    print("🛑 PLAN C (Tüm Ayna Sunucular) başarısız oldu.")
    return []

# ===================================================
# 🔹 PLAN D: YOUTUBE HTML PARSE (COOKIE YOK)
# ===================================================
def get_plan_d_captions(video_id):
    print(f"🧠 [PLAN D] YouTube HTML parse yöntemi deneniyor... ({video_id})")

    url = f"https://www.youtube.com/watch?v={video_id}"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
        "Referer": "https://www.google.com/"
    }

    try:
        html = requests.get(url, headers=headers, timeout=10).text

        match = re.search(r"ytInitialPlayerResponse\s*=\s*({.+?});", html)
        if not match:
            print("❌ Player response bulunamadı")
            return []

        data = json.loads(match.group(1))

        tracks = data.get("captions", {}).get("playerCaptionsTracklistRenderer", {}).get("captionTracks", [])
        if not tracks:
            print("❌ Altyazı yok")
            return []

        track = None
        for t in tracks:
            if t.get("languageCode") == "tr":
                track = t
                break
        if not track:
            track = tracks[0]

        caption_url = track.get("baseUrl")
        if not caption_url:
            return []

        caption_url += "&fmt=json3"

        res = requests.get(caption_url, headers=headers, timeout=10)
        data = res.json()

        captions = []
        for event in data.get("events", []):
            if "segs" in event:
                text = "".join(seg.get("utf8", "") for seg in event["segs"]).strip()
                if text:
                    start = event.get("tStartMs", 0) / 1000
                    duration = event.get("dDurationMs", 0) / 1000
                    captions.append({
                        "text": text,
                        "start": round(start, 2),
                        "end": round(start + duration, 2)
                    })

        if captions:
            print(f"✅ PLAN D Başarılı: {len(captions)} satır çekildi.")
            return captions

    except Exception as e:
        print(f"⚠️ PLAN D hata verdi: {e}")

    return []
# ===================================================
# 🔹 ANA ÇEKİCİ (3 MOTORU DA SIRAYLA DENER)
# ===================================================
def get_caption_with_yta(video_id: str):
    print(f"🔍 [PLAN A] youtube-transcript-api deneniyor... ({video_id})")
    captions = get_yta_captions(video_id)
    if captions:
        print(f"✅ PLAN A Başarılı: {len(captions)} satır çekildi.")
        return captions

    print(f"⚠️ PLAN A İşe Yaramadı. 🔍 [PLAN B] yt-dlp deneniyor...")
    captions = get_ytdlp_captions(video_id)
    if captions:
        print(f"✅ PLAN B Başarılı: {len(captions)} satır çekildi.")
        return captions

    print(f"⚠️ PLAN B İşe Yaramadı. 🌍 [PLAN C] piped deneniyor...")
    captions = get_plan_c_captions(video_id)
    if captions:
        print(f"✅ PLAN C Başarılı.")
        return captions

    print(f"⚠️ PLAN C İşe Yaramadı. 🧠 [PLAN D] HTML parse deneniyor...")
    captions = get_plan_d_captions(video_id)
    if captions:
        return captions

    print("🛑 TÜM PLANLAR FAIL (IP BAN olabilir)")
    return []


# ===================================================
# 🔹 TAHMİN FONKSİYONLARI VE ANA ANALİZ
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

    print("⏳ 3 /3: Linear SVC Modeli RAM'e yükleniyor...")
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
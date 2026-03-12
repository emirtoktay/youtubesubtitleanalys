import re
import threading  # Eşzamanlı istekleri engellemek için
from flask import Flask, request, jsonify
from flask_cors import CORS

# SADECE GEREKLİ MODÜLLER KALDI (Görsel analizler silindi)
import subtitle_analyzer
import db_manager

app = Flask(__name__)
CORS(app)

# ===================================================
# 🔹 KİLİT MEKANİZMASI (Aynı anda aynı videoyu analiz etmeyi önler)
# ===================================================
active_analyses = set()
analysis_lock = threading.Lock()


# ===================================================
# 🔹 YARDIMCI FONKSİYONLAR
# ===================================================
def extract_video_id(link: str):
    pattern = r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})"
    match = re.search(pattern, link)
    if match: return match.group(1)
    raise ValueError("Geçersiz YouTube bağlantısı.")


def get_canonical_url(video_id: str):
    return f"https://www.youtube.com/watch?v={video_id}"


def calculate_age_rating(text_scores, visual_scores):
    # ---------------------------------------------------------
    # 1. METİN (ALTYAZI) ANALİZİ YAŞ HESAPLAMASI
    # ---------------------------------------------------------
    lstm = text_scores.get('lstm', 100.0)
    bert = text_scores.get('bert', 100.0)
    svc = text_scores.get('svc', 100.0)

    t_scores = [lstm, bert, svc]

    if sum(1 for s in t_scores if s >= 90) >= 2:
        text_age = 7
    elif sum(1 for s in t_scores if s >= 85) >= 2:
        text_age = 9
    elif sum(1 for s in t_scores if s >= 75) >= 2:
        text_age = 13
    elif sum(1 for s in t_scores if s >= 60) >= 2:
        text_age = 15
    else:
        text_age = 18

    # ---------------------------------------------------------
    # 2. GÖRSEL ANALİZ YAŞ HESAPLAMASI (ŞU AN DEVRE DIŞI - HEP 7 DÖNECEK)
    # ---------------------------------------------------------
    visual_age = 7  # Görsel analiz iptal edildiği için varsayılan güvenli değer

    # ---------------------------------------------------------
    # 3. FİNAL YAŞ SINIRI KARARI
    # ---------------------------------------------------------
    final_age = max(text_age, visual_age)

    if final_age == 18:
        return "+18"
    elif final_age == 15:
        return "+15"
    elif final_age == 13:
        return "+13"
    elif final_age == 9:
        return "+9"
    else:
        return "Genel İzleyici (7+)"


# ===================================================
# 🔹 API UÇ NOKTASI (SADECE ALTYAZI)
# ===================================================
@app.route('/analyze_youtube', methods=['POST'])
def analyze_youtube():
    data = request.get_json()
    link = data.get('youtube_link')

    if not link:
        return jsonify({"error": "youtube_link parametresi gerekli."}), 400

    try:
        video_id = extract_video_id(link)
        canonical_url = get_canonical_url(video_id)

        # 1. DB KONTROLÜ
        cached_result = db_manager.check_db_for_result(canonical_url)
        if cached_result:
            text_scores = cached_result.get('safety_percentages', {})
            # Cache'den gelen görsel veriler veya boş dict
            visual_scores = cached_result.get('safety_percentages', {}).get('visual', {})
            age_rating = calculate_age_rating(text_scores, visual_scores)
            cached_result['age_rating'] = age_rating
            return jsonify(cached_result)

        # ----------------------------------------------------
        # 🛡️ KİLİT MEKANİZMASI KONTROLÜ
        # ----------------------------------------------------
        with analysis_lock:
            if video_id in active_analyses:
                return jsonify({"status": "processing", "message": "Bu video zaten şu an analiz ediliyor."})
            active_analyses.add(video_id)

        try:
            # --- 2. SADECE METİN ANALİZİ (GÖRSEL İPTAL) ---
            print(f"📝 Sadece Altyazı analizi yapılıyor... ({video_id})")
            sub_results = subtitle_analyzer.analyze_subtitles(video_id)

            if sub_results:
                text_percentages = sub_results["percentages"]
                total_lines = sub_results["total_lines"]
            else:
                text_percentages = {"lstm": 100.0, "bert": 100.0, "svc": 100.0}
                total_lines = 0

            # Görsel analiz iptal edildiği için veritabanına gönderilecek sahte "temiz" veriler
            visual_results = {
                "gun_safety": 100.0, "knife_safety": 100.0, "gun_det": 0, "knife_det": 0,
                "combined_gun_safety": 100.0, "combined_knife_safety": 100.0,
                "combined_gun_det": 0, "combined_knife_det": 0,
                "gambling_safety": 100.0, "gambling_det": 0
            }

            # --- 3. DB'ye KAYDETME ---
            try:
                db_manager.save_result_to_db(canonical_url, video_id, total_lines, text_percentages, visual_results)
            except Exception as db_err:
                print(f"❌ DB yazma hatası: {db_err}")

            age_rating = calculate_age_rating(text_percentages, visual_results)

            # Eklentiye giden nihai cevap
            return jsonify({
                "status": "success",
                "total_lines": total_lines,
                "safety_percentages": text_percentages,
                "age_rating": age_rating
            })

        finally:
            # 🔓 İŞLEM BİTTİĞİNDE KİLİDİ AÇ
            with analysis_lock:
                if video_id in active_analyses:
                    active_analyses.remove(video_id)

    except Exception as e:
        print(f"❌ Genel Hata: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Railway'in atadığı portu al, yoksa 8080 kullan
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
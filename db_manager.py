import os
import pg8000.native


def get_db_connection():
    try:
        db_url = os.getenv("DATABASE_URL")
        if not db_url: return None

        # URL'yi parçalayıp pg8000'e veriyoruz
        url_parts = db_url.replace("postgresql://", "").split("@")
        user_pass = url_parts[0].split(":")
        host_port_db = url_parts[1].split("/")
        host_port = host_port_db[0].split(":")

        conn = pg8000.native.Connection(
            user=user_pass[0],
            password=user_pass[1] if len(user_pass) > 1 else "",
            host=host_port[0],
            port=int(host_port[1]) if len(host_port) > 1 else 5432,
            database=host_port_db[1],
            timeout=5
        )
        return conn
    except Exception as e:
        print(f"⚠️ Veritabanı bağlantı hatası: {e}")
        return None


# CHECK ve SAVE fonksiyonlarındaki Execute kısımlarını pg8000'e uygun hale getir:
def check_db_for_result(canonical_url: str):
    conn = get_db_connection()
    if not conn: return None
    try:
        # pg8000 dict yerine liste döner
        row = conn.run(
            """
            SELECT TotalLines, LSTM_Safety_Percent, BERT_Safety_Percent, SVC_Safety_Percent, 
                   Gun_Safety_Percent, Knife_Safety_Percent, GunDetections, KnifeDetections,
                   Gun_Safety_Percent_Combined, Knife_Safety_Percent_Combined, 
                   GunDetectionsCombined, KnifeDetectionsCombined, 
                   Gambling_Safety_Percent, GamblingDetections, AnalysisDate 
            FROM AnalysisResults WHERE VideoURL = :url
            """,
            url=canonical_url
        )
        if row:
            r = row[0]  # İlk sonucu al
            return {
                "status": "cached",
                "total_lines": r[0],
                "safety_percentages": {
                    "lstm": float(r[1]), "bert": float(r[2]), "svc": float(r[3]),
                    "visual": {
                        "gun_safety": float(r[4]) if r[4] is not None else 100.0,
                        "knife_safety": float(r[5]) if r[5] is not None else 100.0,
                        "gun_det": int(r[6]) if r[6] is not None else 0,
                        "knife_det": int(r[7]) if r[7] is not None else 0,
                        "combined_gun_safety": float(r[8]) if r[8] is not None else 100.0,
                        "combined_knife_safety": float(r[9]) if r[9] is not None else 100.0,
                        "combined_gun_det": int(r[10]) if r[10] is not None else 0,
                        "combined_knife_det": int(r[11]) if r[11] is not None else 0,
                        "gambling_safety": float(r[12]) if r[12] is not None else 100.0,
                        "gambling_det": int(r[13]) if r[13] is not None else 0
                    }
                },
                "analysis_date": str(r[14])
            }
        return None
    except Exception as e:
        print(f"⚠️ DB okuma hatası: {e}")
        return None
    finally:
        if conn: conn.close()


def save_result_to_db(canonical_url, video_id, total_lines, text_p, visual_p=None):
    conn = get_db_connection()
    if not conn: return
    if visual_p is None:
        visual_p = {
            'gun_safety': 100.0, 'knife_safety': 100.0, 'gun_det': 0, 'knife_det': 0,
            'combined_gun_safety': 100.0, 'combined_knife_safety': 100.0,
            'combined_gun_det': 0, 'combined_knife_det': 0,
            'gambling_safety': 100.0, 'gambling_det': 0
        }
    try:
        conn.run(
            """
            INSERT INTO AnalysisResults 
            (VideoURL, VideoID, TotalLines, LSTM_Safety_Percent, BERT_Safety_Percent, SVC_Safety_Percent, 
             AnalysisDate, VisualAnalysisDate,
             Gun_Safety_Percent, Knife_Safety_Percent, GunDetections, KnifeDetections,
             Gun_Safety_Percent_Combined, Knife_Safety_Percent_Combined, 
             GunDetectionsCombined, KnifeDetectionsCombined,
             Gambling_Safety_Percent, GamblingDetections)
            VALUES (:url, :vid, :tl, :lstm, :bert, :svc, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 
                    :gs, :ks, :gd, :kd, :cgs, :cks, :cgd, :ckd, :gbs, :gbd)
            """,
            url=canonical_url, vid=video_id, tl=total_lines,
            lstm=text_p['lstm'], bert=text_p['bert'], svc=text_p['svc'],
            gs=visual_p['gun_safety'], ks=visual_p['knife_safety'], gd=visual_p['gun_det'], kd=visual_p['knife_det'],
            cgs=visual_p['combined_gun_safety'], cks=visual_p['combined_knife_safety'],
            cgd=visual_p['combined_gun_det'], ckd=visual_p['combined_knife_det'],
            gbs=visual_p['gambling_safety'], gbd=visual_p['gambling_det']
        )
        print("💾 Analiz sonuçları DB'ye kaydedildi.")
    except Exception as e:
        print(f"⚠️ DB yazma hatası: {e}")
    finally:
        if conn: conn.close()
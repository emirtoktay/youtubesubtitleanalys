import psycopg2
import os


def get_db_connection():
    try:
        # 💡 DİKKAT: Kendi bilgisayarında test ederken 'sifren' yazan yeri pgAdmin şifrenle değiştir!
        # Railway'e attığımızda os.getenv("DATABASE_URL") sayesinde otomatik olarak Railway'in DB'sine bağlanacak.
        db_url = os.getenv("DATABASE_URL", "degisken")

        conn = psycopg2.connect(db_url, connect_timeout=3)
        return conn
    except Exception as e:
        print(f"⚠️ PostgreSQL bağlantı hatası (Veritabanı kapalı olabilir): {e}")
        return None


def check_db_for_result(canonical_url: str):
    conn = get_db_connection()
    if not conn: return None

    try:
        cursor = conn.cursor()
        # PostgreSQL'de parametreler '?' yerine '%s' ile belirtilir.
        cursor.execute(
            """
            SELECT TotalLines, LSTM_Safety_Percent, BERT_Safety_Percent, SVC_Safety_Percent, 
                   Gun_Safety_Percent, Knife_Safety_Percent, GunDetections, KnifeDetections,
                   Gun_Safety_Percent_Combined, Knife_Safety_Percent_Combined, 
                   GunDetectionsCombined, KnifeDetectionsCombined, 
                   Gambling_Safety_Percent, GamblingDetections, AnalysisDate 
            FROM AnalysisResults WHERE VideoURL = %s
            """,
            (canonical_url,)
        )
        row = cursor.fetchone()
        if row:
            return {
                "status": "cached",
                "total_lines": row[0],
                "safety_percentages": {
                    "lstm": float(row[1]), "bert": float(row[2]), "svc": float(row[3]),
                    "visual": {
                        "gun_safety": float(row[4]) if row[4] is not None else 100.0,
                        "knife_safety": float(row[5]) if row[5] is not None else 100.0,
                        "gun_det": int(row[6]) if row[6] is not None else 0,
                        "knife_det": int(row[7]) if row[7] is not None else 0,

                        "combined_gun_safety": float(row[8]) if row[8] is not None else 100.0,
                        "combined_knife_safety": float(row[9]) if row[9] is not None else 100.0,
                        "combined_gun_det": int(row[10]) if row[10] is not None else 0,
                        "combined_knife_det": int(row[11]) if row[11] is not None else 0,

                        "gambling_safety": float(row[12]) if row[12] is not None else 100.0,
                        "gambling_det": int(row[13]) if row[13] is not None else 0
                    }
                },
                "analysis_date": str(row[14])
            }
        return None
    except Exception as e:
        print(f"⚠️ DB okuma hatası: {e}")
        return None
    finally:
        if conn:
            cursor.close()
            conn.close()


def save_result_to_db(canonical_url, video_id, total_lines, text_p, visual_p=None):
    conn = get_db_connection()
    if not conn:
        print("💾 Veritabanı yok, analiz sonucu sadece eklentiye gönderilecek (Kaydedilmedi).")
        return

    # Sadece altyazı analizi yapacağımız için, görsel veriler hata vermesin diye sahte (temiz) veri yolluyoruz
    if visual_p is None:
        visual_p = {
            'gun_safety': 100.0, 'knife_safety': 100.0, 'gun_det': 0, 'knife_det': 0,
            'combined_gun_safety': 100.0, 'combined_knife_safety': 100.0,
            'combined_gun_det': 0, 'combined_knife_det': 0,
            'gambling_safety': 100.0, 'gambling_det': 0
        }

    try:
        cursor = conn.cursor()
        # PostgreSQL'de GETDATE() yerine CURRENT_TIMESTAMP kullanılır.
        cursor.execute(
            """
            INSERT INTO AnalysisResults 
            (VideoURL, VideoID, TotalLines, LSTM_Safety_Percent, BERT_Safety_Percent, SVC_Safety_Percent, 
             AnalysisDate, VisualAnalysisDate,
             Gun_Safety_Percent, Knife_Safety_Percent, GunDetections, KnifeDetections,
             Gun_Safety_Percent_Combined, Knife_Safety_Percent_Combined, 
             GunDetectionsCombined, KnifeDetectionsCombined,
             Gambling_Safety_Percent, GamblingDetections)
            VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                canonical_url, video_id, total_lines,
                text_p['lstm'], text_p['bert'], text_p['svc'],

                visual_p['gun_safety'], visual_p['knife_safety'],
                visual_p['gun_det'], visual_p['knife_det'],

                visual_p['combined_gun_safety'], visual_p['combined_knife_safety'],
                visual_p['combined_gun_det'], visual_p['combined_knife_det'],

                visual_p['gambling_safety'], visual_p['gambling_det']
            )
        )
        conn.commit()
        print("💾 Analiz sonuçları PostgreSQL DB'ye başarıyla kaydedildi.")
    except Exception as e:
        print(f"⚠️ DB yazma hatası: {e}")
        conn.rollback()
    finally:
        if conn:
            cursor.close()
            conn.close()
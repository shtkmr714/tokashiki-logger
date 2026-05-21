"""
tokashiki_logger.py
毎日8:15 JSTに実行。
渡嘉敷フェリーポータル（tokashiki-ferry.jp）から
マリンライナーとかしき・フェリーとかしきの運航情報を取得し、
Open-Meteo海洋データとともにGoogle Sheetsに記録する。

GitHub Actions から呼び出す。

便別運航状況（座間味の operation_logger.py と同じ設計）:
  マリンライナーとかしき: marine_bin1（泊港発）, marine_bin2（渡嘉敷港発）
  フェリーとかしき:       ferry_bin1（泊港発 10:00）, ferry_bin2（渡嘉敷港発 16:00）
  ferry_turnaround:       折り返し運航フラグ（1=折り返しのみ）
"""

import os
import re
import json
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from bs4 import BeautifulSoup

JST = ZoneInfo("Asia/Tokyo")

# 渡嘉敷航路の気象データ取得ポイント
# 那覇泊港（26.22, 127.68）〜渡嘉敷（26.19, 127.36）の中間・慶良間海峡の外洋部
# ※座間味と同じ慶良間海峡を通るため、ほぼ同一の気象条件
ROUTE_LAT = 26.20
ROUTE_LON = 127.45

TOKASHIKI_PORTAL_URL = "https://tokashiki-ferry.jp/Senpaku/portal"

SHEET_NAME = "tokashiki_operation_log"

SHEET_HEADERS = [
    "date", "recorded_at",
    # マリンライナーとかしき（高速船）便別運航状況（1=運航, 0=欠航/運休）
    "marine_bin1_operated",  # 泊港発
    "marine_bin2_operated",  # 渡嘉敷港発
    "marine_cancel_reason",  # weather/dock/equipment/none
    # フェリーとかしき 便別運航状況
    "ferry_bin1_operated",   # 泊港発（〜10:00）
    "ferry_bin2_operated",   # 渡嘉敷港発（〜16:00）
    "ferry_turnaround",      # 折り返し運航フラグ（0/1）
    "ferry_cancel_reason",
    # デバッグ用
    "notice_text",           # お知らせ（ドック情報など）
    "raw_status_text",       # スクレイプ生テキスト
    # 気象データ（当日）
    "wave_max",
    "swell_max",
    "wind_max",
    # 気象データ（翌日予報）
    "tmr_wave_max",
    "tmr_swell_max",
    "tmr_wind_max",
    # 気象データ（明後日予報）
    "dayafter_wave_max",
    # 派生
    "marine_am_weather_cancel",  # 泊港発が気象欠航（0/1）
    "marine_pm_weather_cancel",  # 渡嘉敷港発が気象欠航（0/1）
    "ferry_weather_cancel",      # フェリーが1便以上気象欠航（0/1）
]


# ============================================================
# 1. 渡嘉敷フェリーポータル スクレイピング
# ============================================================

def _cancel_reason(text):
    """テキストから欠航理由カテゴリを判定"""
    if any(w in text for w in ["機器", "エンジン", "トラブル", "故障", "点検", "整備"]):
        return "equipment"
    if "ドック" in text or "dock" in text.lower():
        return "dock"
    if any(w in text for w in ["通常", "定刻", "出港"]):
        return "none"
    return "weather"


def _extract_notice_text(soup):
    """
    お知らせセクションのテキストを抽出。
    ドック入り期間などの計画運休情報が含まれる。
    """
    notice_texts = []

    for candidate in soup.find_all(["section", "div", "article", "ul", "li", "p"]):
        classes = " ".join(candidate.get("class", []))
        if any(k in classes.lower() for k in ["notice", "info", "news", "announce", "alert"]):
            text = candidate.get_text(separator=" ", strip=True)
            if any(kw in text for kw in ["ドック", "運休", "欠航", "お知らせ"]):
                notice_texts.append(text[:300])

    for h in soup.find_all(["h1", "h2", "h3", "h4"]):
        if any(kw in h.get_text() for kw in ["お知らせ", "ドック", "運休"]):
            sibling = h.find_next_sibling()
            if sibling:
                notice_texts.append(sibling.get_text(separator=" ", strip=True)[:200])

    return " / ".join(notice_texts) if notice_texts else ""


def get_tokashiki_operation_status():
    """
    渡嘉敷フェリーポータルから運航状況を取得。

    戻り値:
    {
        "marine_bins":          [{"time": "10:00", "operated": 1}, ...],  # 2便
        "ferry_bins":           [{"time": "10:00", "operated": 1}, ...],  # 2便
        "marine_cancel_reason": "weather"/"dock"/"equipment"/"none",
        "ferry_cancel_reason":  ...,
        "ferry_turnaround":     0 or 1,
        "notice_text":          "...",
        "raw_text":             "...",
    }
    """
    result = {
        "marine_bins":          [],
        "ferry_bins":           [],
        "marine_cancel_reason": "none",
        "ferry_cancel_reason":  "none",
        "ferry_turnaround":     0,
        "notice_text":          "",
        "raw_text":             "",
    }

    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(TOKASHIKI_PORTAL_URL, headers=headers, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        full_text = soup.get_text(separator="\n", strip=True)

        # お知らせセクション（ドック情報など）
        notice_text = _extract_notice_text(soup)
        if not notice_text:
            dock_lines = [l.strip() for l in full_text.split("\n")
                          if "ドック" in l and len(l.strip()) > 5]
            notice_text = " / ".join(dock_lines[:3])
        result["notice_text"] = notice_text[:400]
        result["raw_text"]    = full_text[:500]

        # 折り返し運航フラグ
        if "折り返し" in full_text or "折り返し" in notice_text:
            result["ferry_turnaround"] = 1

        # テーブルを船種別に分類
        tables = soup.find_all("table")
        marine_table = None
        ferry_table  = None

        for table in tables:
            context = ""
            caption = table.find("caption")
            if caption:
                context = caption.get_text()
            else:
                thead = table.find("thead")
                if thead:
                    context = thead.get_text()
                prev = table.find_previous(["h1", "h2", "h3", "h4", "p", "div"])
                if prev:
                    context += prev.get_text()

            if "マリンライナー" in context or "高速船" in context:
                marine_table = table
            elif "フェリー" in context:
                ferry_table = table

        # テーブル1つだけ検出された場合の補完
        if len(tables) == 2 and marine_table is None and ferry_table is None:
            marine_table = tables[0]
            ferry_table  = tables[1]

        # パース
        if marine_table or ferry_table:
            if marine_table:
                result["marine_bins"], result["marine_cancel_reason"] = \
                    _parse_table(marine_table, notice_text)
            if ferry_table:
                result["ferry_bins"],  result["ferry_cancel_reason"]  = \
                    _parse_table(ferry_table, notice_text)
        else:
            sections = _split_by_ship(full_text)
            if sections:
                result["marine_bins"], result["marine_cancel_reason"] = \
                    _parse_text_section(sections.get("marine", ""), notice_text)
                result["ferry_bins"],  result["ferry_cancel_reason"]  = \
                    _parse_text_section(sections.get("ferry", ""),  notice_text)

        # 便別データが空の場合はテキスト直接判定（フォールバック）
        if not result["marine_bins"]:
            operated, reason = _judge_from_text(full_text, notice_text, "マリンライナー")
            if operated is not None:
                # 2便とも同じステータスで埋める
                result["marine_bins"]          = [{"time": "便1", "operated": operated},
                                                   {"time": "便2", "operated": operated}]
                result["marine_cancel_reason"] = reason
        if not result["ferry_bins"]:
            operated, reason = _judge_from_text(full_text, notice_text, "フェリーとかしき")
            if operated is not None:
                result["ferry_bins"]          = [{"time": "便1", "operated": operated},
                                                  {"time": "便2", "operated": operated}]
                result["ferry_cancel_reason"] = reason

        # ログ出力
        mb = result["marine_bins"]
        fb = result["ferry_bins"]
        print(f"  [渡嘉敷] マリンライナー: "
              f"bin1={mb[0]['operated'] if len(mb)>0 else '?'} "
              f"bin2={mb[1]['operated'] if len(mb)>1 else '?'} "
              f"({result['marine_cancel_reason']})")
        print(f"  [渡嘉敷] フェリー: "
              f"bin1={fb[0]['operated'] if len(fb)>0 else '?'} "
              f"bin2={fb[1]['operated'] if len(fb)>1 else '?'} "
              f"折り返し={result['ferry_turnaround']} "
              f"({result['ferry_cancel_reason']})")
        if notice_text:
            print(f"  [渡嘉敷] お知らせ: {notice_text[:80]}...")

    except Exception as e:
        print(f"  [警告] 渡嘉敷ポータル取得エラー: {e}")

    return result


def _parse_table(table, notice_text=""):
    """
    テーブル要素から便別運航情報を抽出（最大2行 = 泊港発・渡嘉敷港発）。
    ドック中は時刻が「-」になるため、ステータスのみで状態を判定する。
    """
    bins = []
    cancel_texts = []
    bin_index = 0

    for tr in table.find_all("tr")[1:]:   # ヘッダー行をスキップ
        cells = tr.find_all(["td", "th"])
        if len(cells) < 2:
            continue
        texts = [c.get_text(strip=True) for c in cells]

        time_text   = ""
        status_text = ""
        for t in texts:
            if re.match(r"^\d{1,2}:\d{2}$", t):
                time_text = t
            elif any(kw in t for kw in ["定刻", "出港", "運航", "運休", "欠航", "中止", "ドック"]):
                status_text = t

        if not status_text:
            continue   # ステータス不明な行はスキップ

        # 時刻なし（ドック中）は便番号で代替
        if not time_text:
            time_text = f"便{bin_index + 1}"

        if any(kw in status_text for kw in ["運休", "欠航", "中止", "ドック"]):
            operated = 0
            cancel_texts.append(status_text)
        else:
            operated = 1

        bins.append({"time": time_text, "operated": operated})
        bin_index += 1

    # 欠航理由：テーブルのステータス + お知らせ文を合わせて判定
    combined = " ".join(cancel_texts) + " " + notice_text
    if cancel_texts or any(kw in notice_text for kw in ["ドック", "欠航", "運休"]):
        reason = _cancel_reason(combined)
    else:
        reason = "none"

    return bins, reason


def _split_by_ship(text):
    """ページ全テキストを船種別セクションに分割（テーブル未検出時フォールバック）"""
    sections = {}
    marine_pos = text.find("マリンライナー")
    ferry_pos  = text.find("フェリーとかしき")

    if marine_pos == -1 and ferry_pos == -1:
        return None

    if marine_pos >= 0 and ferry_pos >= 0:
        if marine_pos < ferry_pos:
            sections["marine"] = text[marine_pos:ferry_pos]
            sections["ferry"]  = text[ferry_pos:]
        else:
            sections["ferry"]  = text[ferry_pos:marine_pos]
            sections["marine"] = text[marine_pos:]
    elif marine_pos >= 0:
        sections["marine"] = text[marine_pos:]
    else:
        sections["ferry"] = text[ferry_pos:]

    return sections


def _parse_text_section(text, notice_text=""):
    """テキストセクションから便別情報を抽出（テーブル未検出時フォールバック）"""
    bins = []
    cancel_texts = []
    bin_index = 0

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        status_text = ""
        for kw in ["定刻", "出港", "運航", "運休", "欠航", "中止", "ドック"]:
            if kw in line:
                status_text = line
                break

        if not status_text:
            continue

        time_m = re.search(r"\b(\d{1,2}:\d{2})\b", line)
        time_text = time_m.group(1) if time_m else f"便{bin_index + 1}"

        if any(kw in status_text for kw in ["運休", "欠航", "中止", "ドック"]):
            bins.append({"time": time_text, "operated": 0})
            cancel_texts.append(line)
        else:
            bins.append({"time": time_text, "operated": 1})

        bin_index += 1

    combined = " ".join(cancel_texts) + " " + notice_text
    reason = _cancel_reason(combined) if (cancel_texts or any(kw in notice_text for kw in ["ドック", "欠航"])) else "none"
    return bins, reason


def _judge_from_text(full_text, notice_text, ship_name):
    """
    便別データが取れなかった場合のフォールバック。
    ship_name 近傍テキスト + notice_text から運航状況を推定。
    戻り値: (operated: 1/0/None, cancel_reason)
    """
    pos = full_text.find(ship_name)
    context = (full_text[pos:pos + 300] if pos >= 0 else "") + " " + notice_text

    if any(kw in context for kw in ["運休", "欠航", "中止"]):
        return 0, _cancel_reason(context)
    if any(kw in context for kw in ["定刻", "出港", "通常運航"]):
        return 1, "none"
    return None, "none"


def _get_bin(bins, idx):
    """idx番目のbinのoperated値を返す（存在しない場合はNone）"""
    if idx < len(bins):
        return bins[idx]["operated"]
    return None


# ============================================================
# 2. Open-Meteo 海洋・気象データ取得
# ============================================================

def get_marine_weather_data():
    """Open-Meteo から慶良間海峡の海洋・気象データを取得。"""
    result = {
        "today_max_wave":    None,
        "today_max_swell":   None,
        "today_max_wind":    None,
        "tmr_max_wave":      None,
        "tmr_max_swell":     None,
        "tmr_max_wind":      None,
        "dayafter_max_wave": None,
    }

    try:
        marine_url = (
            f"https://marine-api.open-meteo.com/v1/marine"
            f"?latitude={ROUTE_LAT}&longitude={ROUTE_LON}"
            f"&hourly=wave_height,swell_wave_height"
            f"&timezone=Asia%2FTokyo&forecast_days=3"
        )
        marine_data = requests.get(marine_url, timeout=15).json()

        weather_url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={ROUTE_LAT}&longitude={ROUTE_LON}"
            f"&hourly=wind_speed_10m"
            f"&wind_speed_unit=ms"
            f"&timezone=Asia%2FTokyo&forecast_days=3"
        )
        weather_data = requests.get(weather_url, timeout=15).json()

        now = datetime.now(JST)

        def _daily_max(data, key, delta):
            target = (now + timedelta(days=delta)).strftime("%Y-%m-%d")
            times  = data.get("hourly", {}).get("time", [])
            values = data.get("hourly", {}).get(key, [])
            vals   = [v for t, v in zip(times, values)
                      if t.startswith(target) and v is not None]
            return round(max(vals), 2) if vals else None

        result["today_max_wave"]    = _daily_max(marine_data,  "wave_height",       0)
        result["today_max_swell"]   = _daily_max(marine_data,  "swell_wave_height", 0)
        result["today_max_wind"]    = _daily_max(weather_data, "wind_speed_10m",    0)
        result["tmr_max_wave"]      = _daily_max(marine_data,  "wave_height",       1)
        result["tmr_max_swell"]     = _daily_max(marine_data,  "swell_wave_height", 1)
        result["tmr_max_wind"]      = _daily_max(weather_data, "wind_speed_10m",    1)
        result["dayafter_max_wave"] = _daily_max(marine_data,  "wave_height",       2)

        print(f"  [Open-Meteo] 本日 波高{result['today_max_wave']}m "
              f"うねり{result['today_max_swell']}m 風速{result['today_max_wind']}m/s")
        print(f"  [Open-Meteo] 明日 波高{result['tmr_max_wave']}m "
              f"うねり{result['tmr_max_swell']}m 風速{result['tmr_max_wind']}m/s")

    except Exception as e:
        print(f"  [警告] Open-Meteoデータ取得エラー: {e}")

    return result


# ============================================================
# 3. Google Sheets への書き込み
# ============================================================

def log_daily_record():
    """メイン関数。データを収集してGoogle Sheetsに1行追加する。"""
    sheets_id = os.environ.get("GOOGLE_SHEETS_ID_TOKASHIKI")
    svc_json  = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")

    if not sheets_id or not svc_json:
        print("  [スキップ] 環境変数未設定（GOOGLE_SHEETS_ID_TOKASHIKI / GOOGLE_SERVICE_ACCOUNT_JSON）")
        return

    try:
        import gspread
        from google.oauth2.service_account import Credentials
        creds = Credentials.from_service_account_info(
            json.loads(svc_json),
            scopes=["https://spreadsheets.google.com/feeds",
                    "https://www.googleapis.com/auth/drive"]
        )
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(sheets_id)

        try:
            ws = sh.worksheet(SHEET_NAME)
        except Exception:
            ws = sh.add_worksheet(title=SHEET_NAME, rows=5000, cols=len(SHEET_HEADERS))
            ws.append_row(SHEET_HEADERS)
            print(f"  新規シート作成: {SHEET_NAME}")

    except Exception as e:
        print(f"  [エラー] Sheets接続失敗: {e}")
        return

    now = datetime.now(JST)
    today_str = now.strftime("%Y-%m-%d")

    try:
        existing = ws.col_values(1)
        if today_str in existing:
            print(f"  [スキップ] {today_str} の記録はすでに存在します")
            return
    except Exception as e:
        print(f"  [警告] 重複チェックエラー（続行）: {e}")

    print(f"\n[渡嘉敷ロガー] データ収集中（{today_str}）...")

    op      = get_tokashiki_operation_status()
    weather = get_marine_weather_data()

    mb1 = _get_bin(op["marine_bins"], 0)  # マリンライナー 泊港発
    mb2 = _get_bin(op["marine_bins"], 1)  # マリンライナー 渡嘉敷港発
    fb1 = _get_bin(op["ferry_bins"],  0)  # フェリー 泊港発
    fb2 = _get_bin(op["ferry_bins"],  1)  # フェリー 渡嘉敷港発

    marine_am_w_cancel = 1 if mb1 == 0 and op["marine_cancel_reason"] == "weather" else 0
    marine_pm_w_cancel = 1 if mb2 == 0 and op["marine_cancel_reason"] == "weather" else 0
    ferry_w_cancel     = 1 if (fb1 == 0 or fb2 == 0) and op["ferry_cancel_reason"] == "weather" else 0

    row = [
        today_str,
        now.strftime("%Y-%m-%d %H:%M"),
        mb1, mb2,
        op["marine_cancel_reason"],
        fb1, fb2,
        op["ferry_turnaround"],
        op["ferry_cancel_reason"],
        op["notice_text"][:400],
        op["raw_text"][:300],
        weather["today_max_wave"],
        weather["today_max_swell"],
        weather["today_max_wind"],
        weather["tmr_max_wave"],
        weather["tmr_max_swell"],
        weather["tmr_max_wind"],
        weather["dayafter_max_wave"],
        marine_am_w_cancel,
        marine_pm_w_cancel,
        ferry_w_cancel,
    ]

    try:
        ws.append_row(row, value_input_option="USER_ENTERED")
        print(f"  ✅ Sheets記録完了: {today_str}")
        print(f"     マリンライナー bin1={mb1} bin2={mb2} ({op['marine_cancel_reason']})")
        print(f"     フェリー       bin1={fb1} bin2={fb2} 折り返し={op['ferry_turnaround']} ({op['ferry_cancel_reason']})")
    except Exception as e:
        print(f"  [エラー] Sheets書き込み失敗: {e}")


# ============================================================
# エントリーポイント
# ============================================================

if __name__ == "__main__":
    print("=" * 50)
    print(f"Tokashiki Ferry Logger: {datetime.now(JST).strftime('%Y-%m-%d %H:%M')}")
    print("=" * 50)
    log_daily_record()
    print("\n完了。")

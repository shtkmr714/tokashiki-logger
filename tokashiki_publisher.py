"""
tokashiki_publisher.py
渡嘉敷航路（マリンライナーとかしき / フェリーとかしき）の
欠航リスク予報をInstagramに投稿する。
tokashiki_logger.py の log_daily_record() から呼び出す。

リスク判定: 座間味（ferry-forecast）と同じスコア式を使用
  score = wave*0.35 + swell*0.30 + wind*0.20 + warning*0.15
  高速船%: sigmoid(score, inflection=0.42, steepness=14)
  フェリー%: sigmoid(score, inflection=0.52, steepness=12)

投稿: 3枚カルーセル
  1枚目: 短期予報（明日・明後日 / 高速船・フェリー）
  2枚目: 長期予報（3〜7日先）
  3枚目: 予報根拠データ（JMA + 数値予測 + 情報源）
"""

import os
import json
import math
import time
import base64
import traceback
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from PIL import Image, ImageDraw, ImageFont

JST = ZoneInfo("Asia/Tokyo")

# ============================================================
# 設定: 座間味と同じスコア式
# ============================================================

SCORE_WEIGHTS = {
    "wave":    0.35,
    "swell":   0.30,
    "wind":    0.20,
    "warning": 0.15,
}

# 渡嘉敷航路代表点（tokashiki_logger.py と同じ3点、最悪値を採用）
ROUTE_POINTS = [
    (26.198, 127.37),   # 渡嘉敷港沖
    (26.205, 127.52),   # 海峡中央
    (26.21,  127.60),   # 泊沖
]

IMG_SIZE = (1080, 1080)

# ============================================================
# フォント
# ============================================================

def _find_noto_font(weights):
    search_dirs = [
        "/usr/share/fonts/opentype/noto", "/usr/share/fonts/noto-cjk",
        "/usr/share/fonts/truetype/noto", "/usr/share/fonts/noto",
        "/usr/local/share/fonts/noto",    "/usr/share/fonts/opentype",
        "/usr/share/fonts/truetype",
    ]
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for w in weights:
            for ext in [".ttc", ".otf", ".ttf"]:
                p = os.path.join(d, f"NotoSansCJK-{w}{ext}")
                if os.path.exists(p):
                    return p
    try:
        import subprocess
        out = subprocess.check_output(
            ["fc-list", ":lang=ja", "--format=%{file}\n"],
            text=True, timeout=5, stderr=subprocess.DEVNULL
        )
        for line in out.splitlines():
            line = line.strip()
            if line and "Noto" in line and "Sans" in line:
                return line
    except Exception:
        pass
    # Windows ローカル開発用フォールバック（本番Linuxでは上のNotoが優先される）
    win_fonts = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
    for cand in ("YuGothB.ttc", "YuGothM.ttc", "meiryob.ttc", "meiryo.ttc",
                 "msgothic.ttc", "yumin.ttf"):
        p = os.path.join(win_fonts, cand)
        if os.path.exists(p):
            return p
    return None


FONT_REGULAR = _find_noto_font(["Regular"])
FONT_BOLD    = _find_noto_font(["Black", "Bold"])
FONT_MEDIUM  = _find_noto_font(["Medium", "Regular"])


def _load_font(path, size):
    if path:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


# カルーセル統一サイズ（正方形 1254²）。短期＝テンプレ native 1254、
# 長期＝1254直接描画、気象データ＝1080描画→保存時に1254²へ拡大。
OUTPUT_SIZE = (1254, 1254)

# 同梱フォント（画像デザイン仕様 §フォント）。数字%=Manrope Bold / 英語=Inter Medium。
_FONT_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "fonts")
FONT_MANROPE = os.path.join(_FONT_DIR, "Manrope-var.ttf")
FONT_INTER   = os.path.join(_FONT_DIR, "Inter-var.ttf")


def _load_var_font(path, size, instance):
    """可変フォントを名前付きインスタンス（'Bold' / 'Medium' 等）でロード。"""
    try:
        fnt = ImageFont.truetype(path, size)
        try:
            fnt.set_variation_by_name(instance)
        except Exception:
            try:
                fnt.set_variation_by_name(instance.encode())
            except Exception:
                pass
        return fnt
    except Exception:
        return _load_font(None, size)


# ============================================================
# カラー（ferry-forecast と同じ閾値）
# ============================================================

def _get_bg_color(pct):
    if pct is None or pct <= 30:  return "#2E7D32"
    elif pct <= 60:               return "#F9A825"
    elif pct <= 80:               return "#E65100"
    else:                         return "#B71C1C"


def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


# ============================================================
# リスクスコア計算（座間味と同じ式）
# ============================================================

def _calc_score(wave, swell, wind, has_warning=False,
                swell_period=None, gust=None):
    """
    0〜1 のスコアを返す。座間味（ferry_alert.py）と完全に同じ式。
    - うねり周期補正: 8秒超で最大1.3倍
    - 突風補正: 平均風速と突風（25m/s正規化）の大きい方を採用
    """
    wave_score  = min((wave  or 0) / 5.0,  1.0)
    swell_score = min((swell or 0) / 4.0,  1.0)

    # うねり周期補正（長周期うねりは実害が大きい）
    if swell_period and swell:
        factor = 1.0 + max(0.0, swell_period - 8.0) * 0.0375
        swell_score = min(swell_score * min(factor, 1.3), 1.0)

    wind_score  = min((wind  or 0) / 20.0, 1.0)
    # 突風補正（瞬間風速が高い場合は突風ベースで評価）
    if gust:
        wind_score = max(wind_score, min(gust / 25.0, 1.0))

    warning_score = 1.0 if has_warning else 0.0
    return round(
        wave_score  * SCORE_WEIGHTS["wave"]    +
        swell_score * SCORE_WEIGHTS["swell"]   +
        wind_score  * SCORE_WEIGHTS["wind"]    +
        warning_score * SCORE_WEIGHTS["warning"], 3
    )


def _score_to_pct_highspeed(score):
    """座間味と同じ sigmoid: inflection=0.42, steepness=14"""
    pct = 100 / (1 + math.exp(-14.0 * (score - 0.42)))
    return int(round(min(max(pct, 1), 99)))


def _score_to_pct_ferry(score):
    """座間味と同じ sigmoid: inflection=0.52, steepness=12"""
    pct = 100 / (1 + math.exp(-12.0 * (score - 0.52)))
    return int(round(min(max(pct, 1), 99)))


# ============================================================
# JMA データ取得（座間味と同じ area 471000）
# ============================================================

def _get_jma_forecast_waves():
    """
    気象庁forecast JSONから沖縄地方の波高テキスト予報を取得。
    {"今日": "1メートル後2メートル", "明日": "3メートル", "明後日": "4メートル"}
    """
    try:
        url = "https://www.jma.go.jp/bosai/forecast/data/forecast/471000.json"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        result = {}
        for series in data[0].get("timeSeries", []):
            times = series.get("timeDefines", [])
            for area in series.get("areas", []):
                waves = area.get("waves", [])
                if not waves:
                    continue
                area_name = area.get("area", {}).get("name", "")
                if "中南部" not in area_name and "南部" not in area_name:
                    continue
                for i, wave in enumerate(waves):
                    if i < len(times):
                        dt = datetime.fromisoformat(times[i])
                        delta = (dt.date() - datetime.now(JST).date()).days
                        label = {0: "今日", 1: "明日", 2: "明後日"}.get(delta)
                        if label and wave:
                            result[label] = wave
                if result:
                    break
        return result
    except Exception as e:
        print(f"  [警告] JMA波浪予報取得失敗: {e}")
        return {}


def _get_jma_probability():
    """
    気象庁早期注意情報（471000: 沖縄県）から波浪警報級確率を取得。
    {"明日": {"type": "...", "level": "中"}, "明後日": {"type": "...", "level": "なし"}}
    """
    try:
        url = "https://www.jma.go.jp/bosai/probability/data/probability/471000.json"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        result = {}
        for entry in data:
            for series in entry.get("timeSeries", []):
                time_defines = series.get("timeDefines", [])
                for area in series.get("areas", []):
                    if area.get("code") != "471010":
                        continue
                    for i, time_str in enumerate(time_defines):
                        dt = datetime.fromisoformat(time_str)
                        delta = (dt.date() - datetime.now(JST).date()).days
                        label = {1: "明日", 2: "明後日"}.get(delta)
                        if not label:
                            continue
                        for prop in area.get("properties", []):
                            prop_type = prop.get("type", "")
                            if "波浪" not in prop_type and "高波" not in prop_type:
                                continue
                            parts = prop.get("parts", [])
                            if i < len(parts):
                                level = parts[i].get("level", "")
                                if level:
                                    result[label] = {"type": prop_type, "level": level}
        return result
    except Exception as e:
        print(f"  [警告] JMA早期注意情報取得失敗: {e}")
        return {}


_SLACK_ALERT_THRESHOLD = 61  # この%以上で通知


def _send_slack_alert(forecast, now):
    """
    短期＋長期のいずれかで欠航リスクが閾値以上なら Slack に通知。
    内容は欠航可能性%のみ（波高等の気象データは含めない）。
    """
    webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
    if not webhook_url:
        print("  [Slack スキップ] SLACK_WEBHOOK_URL 未設定")
        return

    short   = forecast.get("short_term", [])
    lt      = forecast.get("long_term", {})
    lt_days = lt.get("days", [])

    # 短期＋長期全期間の最大%
    max_all = max(
        [max(d.get("hs_pct") or 0, d.get("fe_pct") or 0) for d in short] +
        [max(d.get("hs_pct") or 0, d.get("fe_pct") or 0) for d in lt_days],
        default=0
    )

    if max_all < _SLACK_ALERT_THRESHOLD:
        print(f"  [Slack スキップ] 全期間最大リスク {max_all}% < {_SLACK_ALERT_THRESHOLD}%")
        return

    DAY_JA = ["（月）","（火）","（水）","（木）","（金）","（土）","（日）"]
    lines  = [
        f"🚨 欠航リスクアラート【渡嘉敷航路】{now.strftime('%-m/%-d %H:%M')}更新",
        "",
    ]
    for d in short:
        dt      = datetime.strptime(d["date"], "%Y-%m-%d")
        max_pct = max(d.get("hs_pct") or 0, d.get("fe_pct") or 0)
        icon    = "🔴" if max_pct >= 81 else ("🟠" if max_pct >= 61 else "🟡")
        hs_str  = f"高速船 {d['hs_pct']}%" if d.get("hs_pct") is not None else "高速船 データなし"
        fe_str  = f"フェリー {d['fe_pct']}%" if d.get("fe_pct") is not None else "フェリー データなし"
        lines.append(f"{icon} {d['label_ja']} {dt.strftime('%-m/%-d')}{DAY_JA[dt.weekday()]}")
        lines.append(f"  {hs_str}  /  {fe_str}")
    lines.append("")
    if lt.get("has_risk"):
        lines.append(f"📅 長期（3〜7日先）  最大 {lt['max_pct']}%  {lt['risk_period']}")
    else:
        lines.append(f"📅 長期（3〜7日先）  懸念なし（最大 {lt.get('max_pct', 0)}%）")
    lines += ["", "⚠️ AI予測・参考値"]

    try:
        resp = requests.post(webhook_url, json={"text": "\n".join(lines)}, timeout=10)
        if resp.status_code == 200:
            print(f"  ✅ Slack アラート送信（最大リスク {max_all}%）")
        else:
            print(f"  [警告] Slack 送信失敗: {resp.status_code}")
    except Exception as e:
        print(f"  [警告] Slack 送信エラー: {e}")


def _load_active_suspensions():
    """
    planned_suspensions.json を読み込み、today <= end のものだけ返す。
    ファイルが存在しない・空の場合は [] を返す。
    """
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "planned_suspensions.json")
    try:
        import json as _json
        with open(json_path, encoding="utf-8") as f:
            all_sus = _json.load(f)
        today = datetime.now(JST).date()
        active = [
            s for s in all_sus
            if s.get("start") and s.get("end")
            and datetime.strptime(s["end"], "%Y-%m-%d").date() >= today
        ]
        if active:
            print(f"  [計画運休] {len(active)}件（期限内）: {[s['vessel_ja'] for s in active]}")
        return active
    except Exception as e:
        print(f"  [警告] planned_suspensions.json 読み込みエラー: {e}")
        return []


def _fmt_wave(text):
    """気象庁波高テキストを整形（メートル→m）"""
    if not text:
        return "—"
    return text.replace("メートル", "m").replace(" ", "")


def _fmt_prob(text):
    """警報級確率テキストを整形"""
    if not text or text in ("なし", ""):
        return "なし / None"
    return text


# ============================================================
# Open-Meteo: 複数地点 batched 取得（最悪値採用）
# ============================================================

def _fetch_forecast(days=8, timeout=60, max_retries=3):
    """
    ROUTE_POINTS の3地点を1リクエストに集約し、
    各日で全地点の最悪値を採用した day_list を返す。
    [{date, max_wave, max_swell, max_wind}, ...]
    """
    lats = ",".join(str(p[0]) for p in ROUTE_POINTS)
    lons = ",".join(str(p[1]) for p in ROUTE_POINTS)
    now  = datetime.now(JST)

    marine_locs  = None
    weather_locs = None

    marine_url = (
        f"https://marine-api.open-meteo.com/v1/marine"
        f"?latitude={lats}&longitude={lons}"
        f"&hourly=wave_height,swell_wave_height,swell_wave_period"
        f"&timezone=Asia%2FTokyo&forecast_days={days}"
    )
    weather_url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lats}&longitude={lons}"
        f"&hourly=wind_speed_10m,wind_gusts_10m&wind_speed_unit=ms"
        f"&timezone=Asia%2FTokyo&forecast_days={days}"
    )

    def _as_list(d):
        return d if isinstance(d, list) else [d]

    for attempt in range(max_retries):
        try:
            if marine_locs is None:
                marine_locs  = _as_list(requests.get(marine_url,  timeout=timeout).json())
            if weather_locs is None:
                weather_locs = _as_list(requests.get(weather_url, timeout=timeout).json())
            break
        except Exception as e:
            print(f"  [警告] Open-Meteo取得エラー (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))

    day_list = []
    for delta in range(days):
        target = (now + timedelta(days=delta)).strftime("%Y-%m-%d")

        def _worst(locs, key, h0=None, h1=None):
            """全地点・指定日の最悪値。h0/h1 指定時はその時間帯[h0,h1)の時刻のみ対象。"""
            best = None
            for loc in (locs or []):
                times_list = loc.get("hourly", {}).get("time", [])
                vals  = loc.get("hourly", {}).get(key, [])
                v = []
                for t, val in zip(times_list, vals):
                    if not t.startswith(target) or val is None:
                        continue
                    if h0 is not None:
                        try:
                            hh = int(t[11:13])
                        except (ValueError, IndexError):
                            continue
                        if not (h0 <= hh < h1):
                            continue
                    v.append(val)
                if v:
                    m = max(v)
                    best = m if best is None else max(best, m)
            return round(best, 2) if best is not None else None

        day_list.append({
            "date":             target,
            "max_wave":         _worst(marine_locs,  "wave_height"),
            # 高速船AM/PM用：運航時間帯（午前6-12 / 午後12-18）の波高最大
            "max_wave_am":      _worst(marine_locs,  "wave_height", 6, 12),
            "max_wave_pm":      _worst(marine_locs,  "wave_height", 12, 18),
            "max_swell":        _worst(marine_locs,  "swell_wave_height"),
            "max_swell_period": _worst(marine_locs,  "swell_wave_period"),
            "max_wind":         _worst(weather_locs, "wind_speed_10m"),
            "max_gust":         _worst(weather_locs, "wind_gusts_10m"),
        })

    return day_list


# ============================================================
# 予報データ構築（座間味と同じ構造の dict を返す）
# ============================================================

def _build_forecast_data(day_list, jma_waves=None, jma_prob=None):
    """
    day_list から構造化予報データを構築。
    short_term（明日・明後日）、long_term（3〜7日先）、weather_data を返す。
    """
    jma_waves = jma_waves or {}
    jma_prob  = jma_prob  or {}
    now = datetime.now(JST)

    # --- short_term ---
    short_term = []
    for delta in [1, 2]:
        d     = day_list[delta] if delta < len(day_list) else {}
        dt    = now + timedelta(days=delta)
        score = _calc_score(d.get("max_wave"), d.get("max_swell"), d.get("max_wind"),
                            swell_period=d.get("max_swell_period"), gust=d.get("max_gust"))
        if d.get("max_wave") is not None:
            hs_pct = _score_to_pct_highspeed(score)
            fe_pct = _score_to_pct_ferry(score)
        else:
            hs_pct = fe_pct = None
        # 高速船AM/PM%：時間帯別波高があればそれを使い、なければ終日値にフォールバック
        def _hs_for_wave(wave):
            if wave is None:
                return hs_pct
            sc = _calc_score(wave, d.get("max_swell"), d.get("max_wind"),
                             swell_period=d.get("max_swell_period"), gust=d.get("max_gust"))
            return _score_to_pct_highspeed(sc)
        hs_am = _hs_for_wave(d.get("max_wave_am"))
        hs_pm = _hs_for_wave(d.get("max_wave_pm"))
        label_ja = "明日" if delta == 1 else "明後日"
        label_en = "Tomorrow" if delta == 1 else "Day After"
        short_term.append({
            "date":           d.get("date", dt.strftime("%Y-%m-%d")),
            "date_label":     dt.strftime("%-m/%-d"),
            "date_label_en":  dt.strftime("%b %-d"),
            "label_ja":       label_ja,
            "label_en":       label_en,
            "hs_pct":         hs_pct,
            "hs_am_pct":      hs_am,
            "hs_pm_pct":      hs_pm,
            "fe_pct":         fe_pct,
            "suspended_highspeed": False,
            "suspended_ferry":     False,
            "jma_wave":       jma_waves.get(label_ja, ""),
            "jma_prob":       jma_prob.get(label_ja, {}).get("level", ""),
            "max_wave":       d.get("max_wave"),
            "max_swell":      d.get("max_swell"),
            "max_wind":       d.get("max_wind"),
        })
        if d.get("max_wave") is not None:
            print(f"  {d.get('date', '')}: 波{d.get('max_wave')}m / 高速船{hs_pct}% / フェリー{fe_pct}%")

    # --- long_term ---
    lt_days = []
    risk_dates = []
    for delta in range(3, 8):
        d  = day_list[delta] if delta < len(day_list) else {}
        dt = now + timedelta(days=delta)
        score = _calc_score(d.get("max_wave"), d.get("max_swell"), d.get("max_wind"),
                            swell_period=d.get("max_swell_period"), gust=d.get("max_gust"))
        if d.get("max_wave") is not None:
            hs_pct = _score_to_pct_highspeed(score)
            fe_pct = _score_to_pct_ferry(score)
        else:
            hs_pct = fe_pct = 1
        lt_days.append({
            "date":       d.get("date", dt.strftime("%Y-%m-%d")),
            "date_label": dt.strftime("%-m/%-d"),
            "hs_pct":     hs_pct,
            "fe_pct":     fe_pct,
            "suspended_highspeed": False,
            "suspended_ferry":     False,
        })
        if max(hs_pct or 0, fe_pct or 0) >= 31:
            risk_dates.append(dt)

    max_lt_pct = max(
        (max(d["hs_pct"] or 0, d["fe_pct"] or 0) for d in lt_days),
        default=0
    )

    if risk_dates:
        long_term = {
            "has_risk":       True,
            "risk_period":    f"{risk_dates[0].strftime('%-m/%-d')}〜{risk_dates[-1].strftime('%-m/%-d')}頃",
            "risk_period_en": f"Around {risk_dates[0].strftime('%b %-d')} - {risk_dates[-1].strftime('%b %-d')}",
            "max_pct":        max_lt_pct,
            "days":           lt_days,
        }
    else:
        if lt_days:
            d0 = datetime.strptime(lt_days[0]["date"], "%Y-%m-%d")
            d1 = datetime.strptime(lt_days[-1]["date"], "%Y-%m-%d")
            lt_period_en = f"{d0.strftime('%b %-d')} - {d1.strftime('%b %-d')}"
            lt_period_ja = f"{d0.strftime('%-m/%-d')}〜{d1.strftime('%-m/%-d')}"
        else:
            lt_period_en = lt_period_ja = ""
        long_term = {
            "has_risk":       False,
            "risk_period":    "懸念なし",
            "risk_period_en": "No concern",
            "lt_period_ja":   lt_period_ja,
            "lt_period_en":   lt_period_en,
            "max_pct":        max_lt_pct,
            "days":           lt_days,
        }

    # --- weather_data ---
    tmr = short_term[0] if short_term else {}
    weather_data = {
        "jma_wave_tomorrow":  _fmt_wave(jma_waves.get("明日", "")),
        "jma_wave_dayafter":  _fmt_wave(jma_waves.get("明後日", "")),
        "jma_prob_tomorrow":  _fmt_prob(jma_prob.get("明日", {}).get("level", "")),
        "jma_prob_dayafter":  _fmt_prob(jma_prob.get("明後日", {}).get("level", "")),
        "num_max_wave":  f"{tmr.get('max_wave')}m"    if tmr.get("max_wave")  else "—",
        "num_max_swell": f"{tmr.get('max_swell')}m"   if tmr.get("max_swell") else "—",
        "num_max_wind":  f"{tmr.get('max_wind')} m/s" if tmr.get("max_wind")  else "—",
    }

    return {
        "short_term":          short_term,
        "long_term":           long_term,
        "weather_data":        weather_data,
        "generated_at":        now.strftime("%Y/%m/%d %H:%M"),
        "generated_at_label":  "8:15更新" if now.hour < 11 else "14:30更新",
        "update_date_ja":      now.strftime("%-m/%-d"),
        "update_date_en":      now.strftime("%b %-d"),
    }


# ============================================================
# 画像生成（テンプレ合成＋可変フィールド方式・座間味と同デザイン）
# ============================================================

# ── 渡嘉敷ルート定数 ──
TEMPLATE_DIR       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "templates")
TOK_SHORT_TEMPLATE = os.path.join(TEMPLATE_DIR, "format_Tokashiki_short.jpg")
TOK_ISLAND         = os.path.join(TEMPLATE_DIR, "island_tokashiki_clean.png")
TOK_CARDS          = [(461, 58, 817, 990), (836, 58, 1202, 990)]   # 短期テンプレ実測カード座標
TOK_LINE_JA, TOK_LINE_EN = "渡嘉敷 ⇔ 那覇", "Tokashiki ⇔ Naha"
TOK_OFFICIAL_JA, TOK_OFFICIAL_EN = "渡嘉敷村HP", "Tokashiki Village website"
_CARD_WHITE = (244, 246, 248)
DAY_JA = ["月", "火", "水", "木", "金", "土", "日"]


def _risk_band(pct):
    """欠航%→5段階(日本語,英語,文字色RGB)。下部 RISK LEVEL GUIDE と一致。"""
    if pct <= 10:  return ("低い", "LOW", (46, 125, 50))
    if pct <= 30:  return ("やや低い", "LOW-MID", (104, 159, 56))
    if pct <= 50:  return ("やや高い", "MID", (240, 160, 0))
    if pct <= 80:  return ("高い", "HIGH", (230, 81, 0))
    return            ("非常に高い", "VERY HIGH", (178, 28, 28))


def _band_tint(color, a=0.14):
    return tuple(int(255 * (1 - a) + c * a) for c in color)


def _eff_max(day):
    """短期：運航中船種の欠航%最大（高速船=max(AM,PM)）。両運休なら0。"""
    cands = []
    if not day.get("suspended_highspeed"):
        cands.append(max(day.get("hs_am_pct") or 0, day.get("hs_pm_pct") or 0))
    if not day.get("suspended_ferry"):
        cands.append(day.get("fe_pct") or 0)
    return max(cands) if cands else 0


def _dashed_rounded_rect(draw, box, radius, color, width=3, dash=13, gap=9):
    x0, y0, x1, y1 = box
    r = radius

    def dash_line(xa, ya, xb, yb):
        L = math.hypot(xb - xa, yb - ya)
        if L == 0:
            return
        ux, uy = (xb - xa) / L, (yb - ya) / L
        dd = 0.0
        while dd < L:
            e = min(dd + dash, L)
            draw.line([(xa + ux*dd, ya + uy*dd), (xa + ux*e, ya + uy*e)], fill=color, width=width)
            dd += dash + gap

    dash_line(x0 + r, y0, x1 - r, y0)
    dash_line(x1, y0 + r, x1, y1 - r)
    dash_line(x1 - r, y1, x0 + r, y1)
    dash_line(x0, y1 - r, x0, y0 + r)
    draw.arc([x0, y0, x0 + 2*r, y0 + 2*r], 180, 270, fill=color, width=width)
    draw.arc([x1 - 2*r, y0, x1, y0 + 2*r], 270, 360, fill=color, width=width)
    draw.arc([x1 - 2*r, y1 - 2*r, x1, y1], 0, 90, fill=color, width=width)
    draw.arc([x0, y1 - 2*r, x0 + 2*r, y1], 90, 180, fill=color, width=width)


def _draw_cancel_icon(draw, cx, cy, r, color):
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    o = int(r * 0.42)
    w = max(3, r // 6)
    draw.line([(cx - o, cy - o), (cx + o, cy + o)], fill="white", width=w)
    draw.line([(cx - o, cy + o), (cx + o, cy - o)], fill="white", width=w)


def _ocean_bg(size):
    w, h = size
    img = Image.new("RGB", size)
    d = ImageDraw.Draw(img)
    top, bot = (6, 124, 190), (1, 58, 116)
    for yy in range(h):
        t = yy / h
        d.line([(0, yy), (w, yy)], fill=tuple(int(top[i]*(1-t) + bot[i]*t) for i in range(3)))
    return img


def _draw_risk_guide(draw, cx, y, fonts):
    items = [
        ("0-10%",   "低い",       "LOW",       (46, 125, 50)),
        ("10-30%",  "やや低い",   "LOW-MID",   (104, 159, 56)),
        ("30-50%",  "やや高い",   "MID",       (240, 160, 0)),
        ("50-80%",  "高い",       "HIGH",      (230, 81, 0)),
        ("80-100%", "非常に高い", "VERY HIGH", (178, 28, 28)),
    ]
    n = len(items)
    span = 1090
    x0 = cx - span // 2
    step = span // n
    for i, (rng, ja, en, col) in enumerate(items):
        ix = x0 + i * step + 14
        draw.ellipse([ix, y - 12, ix + 24, y + 12], fill=col)
        tx = ix + 36
        draw.text((tx, y - 18), rng, font=fonts["g_pct"], fill=(40, 50, 70), anchor="lm")
        draw.text((tx, y + 2),  ja,  font=fonts["g_ja"],  fill=(40, 50, 70), anchor="lm")
        draw.text((tx, y + 20), en,  font=fonts["g_en"],  fill=(120, 130, 150), anchor="lm")


# ── 長期レイアウト定数（正方形1254²固定）──
_LT_W, _LT_CX = OUTPUT_SIZE[0], OUTPUT_SIZE[0] // 2
_LT_ISLAND_BOX = (40, 182, 412, 548)
_LT_SUM_CARD   = (432, 178, 1214, 548)
_LT_BARS_CARD  = (40, 575, 1214, 1045)
_LT_GUIDE_CARD = (40, 1062, 1214, 1148)
_LT_ROWS = 5
_LT_AREA_TOP, _LT_AREA_BOT = 655, 1035
_LT_ROW_H = (_LT_AREA_BOT - _LT_AREA_TOP) // _LT_ROWS
_LT_BAR_H = min(30, _LT_ROW_H - 30)
_LT_COLS = [
    {"rx0": 40,  "key": "hs_pct", "sus": "suspended_highspeed", "lab": ("高速船", "High-speed boat")},
    {"rx0": 627, "key": "fe_pct", "sus": "suspended_ferry",     "lab": ("フェリー", "Ferry")},
]


def _lt_col_xy(col):
    rx0 = col["rx0"]
    return rx0 + 140, rx0 + 158, rx0 + 470, rx0 + 483


def _longterm_fonts():
    return {
        "title":     _load_font(FONT_BOLD,   40),
        "title_en":  _load_var_font(FONT_INTER, 21, "Medium"),
        "route":     _load_font(FONT_MEDIUM, 20),
        "head":      _load_font(FONT_MEDIUM, 26),
        "period":    _load_font(FONT_BOLD,   54),
        "period_en": _load_var_font(FONT_INTER, 22, "Medium"),
        "vlabel":    _load_font(FONT_MEDIUM, 23),
        "vmax":      _load_var_font(FONT_MANROPE, 58, "Bold"),
        "maxlbl":    _load_font(FONT_MEDIUM, 16),
        "colhd":     _load_font(FONT_MEDIUM, 24),
        "date":      _load_font(FONT_MEDIUM, 20),
        "barpct":    _load_var_font(FONT_MANROPE, 24, "Bold"),
        "susp":      _load_font(FONT_MEDIUM, 19),
        "g_pct":     _load_var_font(FONT_INTER, 15, "SemiBold"),
        "g_ja":      _load_font(FONT_MEDIUM, 15),
        "g_en":      _load_var_font(FONT_INTER, 12, "Medium"),
        "xs":        _load_font(FONT_REGULAR, 17),
        "xs_en":     _load_var_font(FONT_INTER, 15, "Medium"),
    }


def _paste_island(img):
    try:
        isl = Image.open(TOK_ISLAND).convert("RGBA")
    except Exception as e:
        print(f"  [警告] 島マップ読込失敗（{e}）→ スキップ")
        return
    bx0, by0, bx1, by1 = _LT_ISLAND_BOX
    aw, ah = bx1 - bx0, by1 - by0
    scale = min(aw / isl.width, ah / isl.height)
    nw, nh = int(isl.width * scale), int(isl.height * scale)
    isl = isl.resize((nw, nh), Image.LANCZOS)
    img.paste(isl, (bx0 + (aw - nw) // 2, by0 + (ah - nh) // 2), isl)


def _render_longterm_static():
    """長期固定背景（ヘッダー・島・カード枠・静的ラベル・軌道・ガイド・免責）を描画。"""
    img = _ocean_bg(OUTPUT_SIZE)
    draw = ImageDraw.Draw(img)
    f = _longterm_fonts()
    cx = _LT_CX
    draw.rectangle([(0, 0), (_LT_W, 150)], fill=(13, 47, 92))
    draw.text((cx, 50), "フェリー欠航可能性 長期予報（3〜7日先）", font=f["title"], fill="white", anchor="mm")
    draw.text((cx, 92), "Ferry Cancellation Risk  /  Long-term Forecast (3-7 days ahead)",
              font=f["title_en"], fill=(200, 220, 245), anchor="mm")
    draw.text((cx, 122), f"{TOK_LINE_JA}   {TOK_LINE_EN}", font=f["route"], fill=(170, 200, 235), anchor="mm")
    _paste_island(img)
    sc = _LT_SUM_CARD
    draw.rounded_rectangle(sc, radius=22, fill=(248, 250, 252))
    scx = (sc[0] + sc[2]) // 2
    draw.text((scx, 213), "欠航リスク期間  Risk Period", font=f["head"], fill=(90, 100, 120), anchor="mm")
    draw.line([(sc[0] + 38, 348), (sc[2] - 38, 348)], fill=(225, 228, 234), width=2)
    draw.line([(scx, 364), (scx, 528)], fill=(225, 228, 234), width=2)
    col_l = sc[0] + int((sc[2] - sc[0]) * 0.27)
    col_r = sc[0] + int((sc[2] - sc[0]) * 0.73)
    draw.text((col_l, 388), "高速船  High-speed boat", font=f["vlabel"], fill=(70, 80, 100), anchor="mm")
    draw.text((col_r, 388), "フェリー  Ferry",          font=f["vlabel"], fill=(70, 80, 100), anchor="mm")
    draw.rounded_rectangle(_LT_BARS_CARD, radius=22, fill=(248, 250, 252))
    for col in _LT_COLS:
        ccx = col["rx0"] + 587 // 2
        draw.text((ccx, 612), f"{col['lab'][0]}  {col['lab'][1]}", font=f["colhd"], fill=(40, 50, 70), anchor="mm")
    draw.line([(cx, 600), (cx, 1035)], fill=(228, 231, 236), width=1)
    for i in range(_LT_ROWS):
        cyr = _LT_AREA_TOP + i * _LT_ROW_H + _LT_ROW_H // 2
        for col in _LT_COLS:
            _, bx0, bx1, _ = _lt_col_xy(col)
            draw.rounded_rectangle([bx0, cyr - _LT_BAR_H // 2, bx1, cyr + _LT_BAR_H // 2],
                                   radius=_LT_BAR_H // 2, fill=(228, 231, 236))
    draw.rounded_rectangle(_LT_GUIDE_CARD, radius=18, fill=(248, 250, 252))
    _draw_risk_guide(draw, cx, 1102, f)
    draw.text((cx, 1178), f"※AI予測・参考値。公式情報は{TOK_OFFICIAL_JA}をご確認ください。",
              font=f["xs"], fill=(225, 235, 248), anchor="mm")
    draw.text((cx, 1204), f"*AI-based estimate. Check {TOK_OFFICIAL_EN} for the latest information.",
              font=f["xs_en"], fill=(190, 210, 235), anchor="mm")
    return img


_LONGTERM_BG_CACHE = {}


def _get_longterm_bg():
    """長期固定背景をプロセス内メモリにキャッシュして再利用（ディスクには焼かない）。"""
    if "tokashiki" not in _LONGTERM_BG_CACHE:
        _LONGTERM_BG_CACHE["tokashiki"] = _render_longterm_static()
    return _LONGTERM_BG_CACHE["tokashiki"]

def make_image_short(forecast, output_path):
    """画像①: 短期予報（テンプレ format_Tokashiki_short.jpg に2カードを合成）。
    左パネル（海・島マップ・タイトル・航路名）と下部リスクガイドはテンプレのまま、
    右の2カード（明日・明後日）に実予測値（高速船AM/PM・フェリー）を描画する。"""
    short = forecast["short_term"]

    try:
        img = Image.open(TOK_SHORT_TEMPLATE).convert("RGB")
    except Exception as e:
        print(f"  [警告] テンプレート読込失敗（{e}）→ 簡易背景で代替")
        img = Image.new("RGB", OUTPUT_SIZE, color=_hex_to_rgb("#0D47A1"))
    draw = ImageDraw.Draw(img)

    def _nj(sz):  return _load_font(FONT_MEDIUM, sz)
    def _num(sz): return _load_var_font(FONT_MANROPE, sz, "Bold")
    def _int(sz): return _load_var_font(FONT_INTER, sz, "Medium")
    f = {
        "badge":    _nj(31), "label_en": _int(27), "big": _num(150), "pct": _num(70),
        "risk_jp":  _nj(29), "risk_en": _int(23), "sub_lbl": _nj(24), "ampm": _int(33),
        "fe_val":   _num(52), "notice": _nj(17), "susp": _nj(38), "susp_en": _int(17),
        "susp_lbl": _nj(21),
    }
    LABEL_GRAY  = (70, 70, 72)
    NOTICE_GRAY = (120, 124, 130)

    def _draw_big_pct(cx, cy, pct, color):
        num = str(pct)
        nb = draw.textbbox((0, 0), num, font=f["big"])
        pb = draw.textbbox((0, 0), "%", font=f["pct"])
        nw, pw = nb[2]-nb[0], pb[2]-pb[0]
        gap = 6
        x0 = cx - (nw + gap + pw) // 2
        draw.text((x0, cy), num, font=f["big"], fill=color, anchor="lm")
        draw.text((x0 + nw + gap, cy + 34), "%", font=f["pct"], fill=color, anchor="lm")

    def _draw_suspended_box(cx, box, vessel_ja, vessel_en):
        bx0, by0, bx1, by1 = box
        draw.rounded_rectangle(box, radius=18, fill=(238, 240, 242))
        _dashed_rounded_rect(draw, box, 18, NOTICE_GRAY, width=3, dash=13, gap=9)
        nb = draw.textbbox((0, 0), "公式発表 Official Notice", font=f["notice"])
        nw = nb[2]-nb[0]
        badge_y = by0 + 24
        draw.rounded_rectangle([(cx-nw//2-16, badge_y-15), (cx+nw//2+16, badge_y+15)],
                               radius=9, fill=NOTICE_GRAY)
        draw.text((cx, badge_y), "公式発表 Official Notice", font=f["notice"], fill="white", anchor="mm")
        mid_y = by0 + (by1 - by0) // 2 + 8
        susp_w = draw.textbbox((0, 0), "運休", font=f["susp"])[2]
        icon_r = 19
        group_w = icon_r*2 + 12 + susp_w
        gx = cx - group_w // 2
        _draw_cancel_icon(draw, gx + icon_r, mid_y, icon_r, (90, 96, 104))
        draw.text((gx + icon_r*2 + 12, mid_y), "運休", font=f["susp"], fill=(60, 64, 70), anchor="lm")
        draw.text((cx, mid_y + 32), "Suspended", font=f["susp_en"], fill=NOTICE_GRAY, anchor="mm")
        draw.text((cx, by1 - 26), f"{vessel_ja}  {vessel_en}",
                  font=f["susp_lbl"], fill=LABEL_GRAY, anchor="mm")

    cards = [(TOK_CARDS[0], short[0] if len(short) > 0 else {}),
             (TOK_CARDS[1], short[1] if len(short) > 1 else {})]

    for (x0, y0, x1, y1), day in cards:
        if not day:
            continue
        cx = (x0 + x1) // 2
        draw.rounded_rectangle([(x0, y0), (x1, y1)], radius=28, fill=_CARD_WHITE)

        sus_hs = day.get("suspended_highspeed", False)
        sus_fe = day.get("suspended_ferry", False)
        head_pct = _eff_max(day)
        band_ja, band_en, band_col = _risk_band(head_pct)
        tint = _band_tint(band_col)

        draw.rounded_rectangle([(cx-92, 88), (cx+92, 137)], radius=12, fill=band_col)
        draw.text((cx, 112), f"{day['label_ja']} {day['date_label']}",
                  font=f["badge"], fill="white", anchor="mm")
        draw.text((cx, 168), day.get("label_en", "").upper(),
                  font=f["label_en"], fill=band_col, anchor="mm")

        _draw_big_pct(cx, 322, head_pct, band_col)
        draw.line([(x0+40, 452), (x1-40, 452)], fill=(214, 216, 220), width=2)
        draw.text((cx, 494), f"欠航リスク：{band_ja}", font=f["risk_jp"], fill=band_col, anchor="mm")
        draw.text((cx, 528), f"{band_en} RISK", font=f["risk_en"], fill=band_col, anchor="mm")

        sb_x0, sb_x1 = x0+26, x1-26
        HS_BOX = (sb_x0, 600, sb_x1, 783)
        FE_BOX = (sb_x0, 795, sb_x1, 980)

        # ── 高速船（AM/PM）──
        if sus_hs:
            _draw_suspended_box(cx, HS_BOX, "高速船", "High-speed boat")
        else:
            draw.rounded_rectangle(HS_BOX, radius=18, fill=tint)
            draw.text((cx, 645), "高速船  High-speed boat",
                      font=f["sub_lbl"], fill=LABEL_GRAY, anchor="mm")
            am = day.get("hs_am_pct", day.get("hs_pct"))
            pm = day.get("hs_pm_pct", day.get("hs_pct"))
            draw.text((cx, 718), f"AM {am}%  /  PM {pm}%",
                      font=f["ampm"], fill=band_col, anchor="mm")

        # ── フェリー ──
        if sus_fe:
            _draw_suspended_box(cx, FE_BOX, "フェリー", "Ferry")
        else:
            draw.rounded_rectangle(FE_BOX, radius=18, fill=tint)
            draw.text((cx, 843), "フェリー  Ferry", font=f["sub_lbl"], fill=LABEL_GRAY, anchor="mm")
            draw.text((cx, 915), f"{day.get('fe_pct')}%", font=f["fe_val"], fill=band_col, anchor="mm")

    img.save(output_path)
    print(f"  画像①保存: {output_path}")


def make_image_longterm(forecast, output_path):
    """画像②: 長期予報（3〜7日先）。正方形1254²・固定背景＋可変オーバーレイ。
    固定部（ヘッダー/島マップ/カード枠/ラベル/軌道/ガイド/免責）はメモリキャッシュ背景を再利用し、
    可変部（リスク期間・船種別最大%・日別バー）のみ毎回描画する。"""
    lt = forecast["long_term"]
    img = _get_longterm_bg().copy()
    draw = ImageDraw.Draw(img)
    f = _longterm_fonts()

    # ── 可変: リスク期間 ──
    sc = _LT_SUM_CARD
    scx = (sc[0] + sc[2]) // 2
    max_band = _risk_band(lt["max_pct"])
    if lt["has_risk"]:
        draw.text((scx, 272), lt["risk_period"].replace("頃", ""),
                  font=f["period"], fill=max_band[2], anchor="mm")
        draw.text((scx, 314), lt["risk_period_en"],
                  font=f["period_en"], fill=(110, 120, 140), anchor="mm")
    else:
        # 「懸念なし No Significant Risk」はf["period"](54px)だとカード幅(432〜1214)を
        # 超えて左右にはみ出し区切り線にも迫るため、カード内に収まるよう自動縮小する。
        text = "懸念なし  No Significant Risk"
        max_w = (sc[2] - sc[0]) - 84   # カード幅から左右マージン
        size = 54
        while size > 34:
            fnt = _load_font(FONT_BOLD, size)
            if draw.textbbox((0, 0), text, font=fnt)[2] <= max_w:
                break
            size -= 2
        draw.text((scx, 288), text, font=_load_font(FONT_BOLD, size),
                  fill=(46, 125, 50), anchor="mm")

    # ── 可変: 船種別 最大%（運休除外）──
    hs_running = [d["hs_pct"] for d in lt["days"] if not d.get("suspended_highspeed")]
    fe_running = [d["fe_pct"] for d in lt["days"] if not d.get("suspended_ferry")]
    hs_max = max(hs_running) if hs_running else None
    fe_max = max(fe_running) if fe_running else None
    col_l = sc[0] + int((sc[2] - sc[0]) * 0.27)
    col_r = sc[0] + int((sc[2] - sc[0]) * 0.73)
    for vx, pct in [(col_l, hs_max), (col_r, fe_max)]:
        if pct is None:
            draw.text((vx, 446), "全日運休", font=f["vlabel"], fill=(120, 124, 130), anchor="mm")
            draw.text((vx, 484), "All days suspended", font=f["maxlbl"], fill=(150, 156, 166), anchor="mm")
        else:
            b = _risk_band(pct)
            draw.text((vx, 442), f"{pct}%", font=f["vmax"], fill=b[2], anchor="mm")
            draw.text((vx, 484), "最大欠航可能性 / Max Risk", font=f["maxlbl"], fill=(140, 150, 165), anchor="mm")

    # ── 可変: 日別バー ──
    days = lt["days"][:_LT_ROWS]
    for i, d in enumerate(days):
        cyr = _LT_AREA_TOP + i * _LT_ROW_H + _LT_ROW_H // 2
        dt = datetime.strptime(d["date"], "%Y-%m-%d")
        label = f"{dt.month}/{dt.day}({DAY_JA[dt.weekday()]})"
        for col in _LT_COLS:
            date_r, bx0, bx1, px = _lt_col_xy(col)
            pct = d[col["key"]] or 0
            is_sus = d.get(col["sus"], False)
            draw.text((date_r, cyr), label, font=f["date"], fill=(60, 70, 90), anchor="rm")
            if is_sus:
                for sx in range(bx0, bx1, 11):
                    draw.line([(sx, cyr - _LT_BAR_H // 2), (sx + _LT_BAR_H, cyr + _LT_BAR_H // 2)],
                              fill=(176, 184, 196), width=2)
                mid = (bx0 + bx1) // 2
                draw.text((mid, cyr), "運休 Suspended", font=f["susp"], fill=(90, 96, 106), anchor="mm")
            else:
                b = _risk_band(pct)
                bw = int((bx1 - bx0) * max(pct, 1) / 100)
                if bw > _LT_BAR_H:
                    draw.rounded_rectangle([bx0, cyr - _LT_BAR_H // 2, bx0 + bw, cyr + _LT_BAR_H // 2],
                                           radius=_LT_BAR_H // 2, fill=b[2])
                draw.text((px, cyr), f"{pct}%", font=f["barpct"], fill=b[2], anchor="lm")

    img.save(output_path)
    print(f"  画像②保存: {output_path}")


def make_image_weatherdata(forecast, output_path):
    """
    画像③: 予報根拠データ（座間味と同じ3セクション構成）
    JMA / 数値予測 / 情報源。短期・長期と同じ青い海グラデ背景＋白カード。
    """
    wd = forecast.get("weather_data", {})
    img = _ocean_bg(IMG_SIZE)
    draw = ImageDraw.Draw(img)
    # 内容は白カードに載せる（濃色文字で可読性を確保）
    draw.rounded_rectangle([(44, 108), (1036, 1002)], radius=22, fill=(250, 251, 252))

    f = {
        "title":    _load_font(FONT_BOLD,    40),
        "sec_hd":   _load_font(FONT_BOLD,    22),
        "label_ja": _load_font(FONT_REGULAR, 20),
        "label_en": _load_font(FONT_REGULAR, 17),
        "value":    _load_font(FONT_MEDIUM,  20),
        "src":      _load_font(FONT_REGULAR, 19),
        "foot":     _load_font(FONT_REGULAR, 17),
    }

    # タイトル（海グラデ上・白文字）
    draw.text((540, 62), "予報根拠データ  /  Forecast Data",
              font=f["title"], fill="white", anchor="mm")

    def section_header(y, ja, en):
        draw.rounded_rectangle([(60, y), (1020, y + 44)], radius=10, fill=(232, 238, 245))
        draw.text((80, y + 22), f"【{ja} / {en}】",
                  font=f["sec_hd"], fill=(40, 60, 100), anchor="lm")
        return y + 60

    def row(y, icon, label_ja, label_en, value):
        draw.text((80, y), f"{icon} {label_ja}", font=f["label_ja"], fill=(38, 48, 68), anchor="lm")
        draw.text((96, y + 24), label_en, font=f["label_en"], fill=(120, 130, 150), anchor="lm")
        draw.text((1010, y + 12), str(value), font=f["value"], fill=(24, 32, 52), anchor="rm")
        return y + 58

    y = 128

    # 【気象庁 / JMA】
    y = section_header(y, "気象庁", "JMA")
    y = row(y, "🌊", "波高予報（明日）", "Wave Height Forecast (Tomorrow)", wd.get("jma_wave_tomorrow", "—"))
    y = row(y, "🌊", "波高予報（明後日）", "Wave Height Forecast (Day After)", wd.get("jma_wave_dayafter", "—"))
    y = row(y, "⚠", "早期注意情報・波浪（明日）", "Early Warning Wave (Tomorrow)", wd.get("jma_prob_tomorrow", "なし / None"))
    y = row(y, "⚠", "早期注意情報・波浪（明後日）", "Early Warning Wave (Day After)", wd.get("jma_prob_dayafter", "なし / None"))

    y += 12

    # 【数値予測 / Numerical Model】
    y = section_header(y, "数値予測", "Numerical Model")
    y = row(y, "📊", "明日 最大波高", "Tomorrow Max Wave Height", wd.get("num_max_wave", "—"))
    y = row(y, "📊", "明日 最大うねり", "Tomorrow Max Swell Height", wd.get("num_max_swell", "—"))
    y = row(y, "💨", "明日 最大風速", "Tomorrow Max Wind Speed", wd.get("num_max_wind", "—"))

    y += 12

    # 【情報源 / Sources】
    y = section_header(y, "情報源", "Sources")
    draw.text((80, y), "気象庁（jma.go.jp）  /  Open-Meteo Marine API",
              font=f["src"], fill=(38, 48, 68), anchor="lm")
    draw.text((80, y + 30), "渡嘉敷村HP（vill.tokashiki.okinawa.jp）",
              font=f["src"], fill=(38, 48, 68), anchor="lm")
    draw.text((80, y + 60), "Tokashiki Village official site / JMA (jma.go.jp)",
              font=f["src"], fill=(120, 130, 150), anchor="lm")

    # フッター（海グラデ上・淡色）
    draw.text((540, 1038), "※欠航判断は船会社・渡嘉敷村が行います。本データはAI予測の参考値です。",
              font=f["foot"], fill=(225, 235, 248), anchor="mm")
    draw.text((540, 1060), "*Cancellation is determined by ferry operators. AI-based estimates for reference only.",
              font=f["foot"], fill=(200, 216, 236), anchor="mm")

    img = img.resize(OUTPUT_SIZE, Image.LANCZOS)   # カルーセル統一: 1254²
    img.save(output_path)
    print(f"  画像③保存: {output_path}")


def _upload_images_to_github(image_paths):
    token = os.environ.get("GITHUB_TOKEN")
    repo  = os.environ.get("GITHUB_REPOSITORY")
    if not token or not repo:
        print("  [スキップ] GITHUB_TOKEN / GITHUB_REPOSITORY 未設定")
        return []

    owner     = repo.split("/")[0]
    repo_name = repo.split("/")[1]
    headers   = {"Authorization": f"token {token}",
                 "Accept": "application/vnd.github.v3+json"}
    urls = []

    for path in image_paths:
        filename = os.path.basename(path)
        try:
            with open(path, "rb") as fh:
                content = base64.b64encode(fh.read()).decode()
            target_path = f"images/{filename}"
            api_url     = f"https://api.github.com/repos/{repo}/contents/{target_path}"
            existing    = requests.get(api_url, headers=headers)
            sha         = existing.json().get("sha") if existing.status_code == 200 else None
            data        = {"message": f"Auto: {filename}", "content": content, "branch": "main"}
            if sha:
                data["sha"] = sha
            resp = requests.put(api_url, json=data, headers=headers)
            if resp.status_code in (200, 201):
                page_url = f"https://{owner}.github.io/{repo_name}/{target_path}"
                urls.append(page_url)
                print(f"    ✅ {page_url}")
            else:
                print(f"    [警告] アップロード失敗 ({filename}): {resp.status_code}")
        except Exception as e:
            print(f"    [警告] アップロードエラー: {e}")

    return urls


_SECRET_ENV_KEYS = (
    "INSTAGRAM_ACCESS_TOKEN",
    "SLACK_WEBHOOK_URL",
    "DISPATCH_TOKEN",
    "GITHUB_TOKEN",
)


def _redact_secrets(text):
    """例外メッセージにはトークン付きURLが含まれうる。Slackは自動マスクしないので自前で伏せる。"""
    out = str(text)
    for key in _SECRET_ENV_KEYS:
        val = os.environ.get(key)
        if val and len(val) >= 8:
            out = out.replace(val, f"***{key}***")
    return out


def _alert_slack(message):
    """障害をSlackに流す。投稿停止に気づけるようにするための最低限の通知。"""
    webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
    if not webhook_url:
        print("  [Slack スキップ] SLACK_WEBHOOK_URL 未設定")
        return
    try:
        requests.post(webhook_url, json={"text": f"🚨 {_redact_secrets(message)}"}, timeout=10)
    except Exception as e:
        print(f"  [警告] Slack 通知エラー: {e}")


class InstagramPostError(Exception):
    """Instagram投稿の失敗。呼び出し側で通知しジョブを失敗させるために送出する。"""


# ============================================================
# 投稿の重複ガード
# ============================================================
# この島は2つの経路で起動される：
#   1) cron-job.org → kucha-ferry-alert → dispatch（8:15頃・14:30）
#   2) このリポジトリ自身の cron（1が動かなかったときの fallback）
# 2は GitHub のスケジューラ遅延で毎朝9時台に発火し、1の直後に同じ内容を
# もう1本投稿していた（2026-07-04〜07-17 は毎朝重複）。時刻ガードは
# workflow_dispatch を通すため、この重複を防げない。
# そこで「その日その枠で投稿済みか」をリポジトリ内の state ファイルで判定する。
# これにより 2 は本来の fallback（1が落ちた時だけ投稿）として機能する。

POST_STATE_PATH = "state/last_post.json"


def _post_slot(now):
    """投稿枠。午後便の判定（now.hour >= 12）と同じ境界で朝／午後を分ける。"""
    return "afternoon" if now.hour >= 12 else "morning"


def _post_key(now):
    return f"{now.strftime('%Y-%m-%d')}_{_post_slot(now)}"


def _post_state_api():
    repo = os.environ.get("GITHUB_REPOSITORY")
    return f"https://api.github.com/repos/{repo}/contents/{POST_STATE_PATH}"


def _post_state_headers():
    token = os.environ.get("GITHUB_TOKEN")
    return {"Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json"}


def _read_post_state():
    """(state_dict, sha) を返す。未作成なら ({}, None)、判定不能なら (None, None)。"""
    if not os.environ.get("GITHUB_TOKEN") or not os.environ.get("GITHUB_REPOSITORY"):
        return None, None
    resp = requests.get(_post_state_api(), headers=_post_state_headers(), timeout=15)
    if resp.status_code == 404:
        return {}, None
    if resp.status_code != 200:
        raise RuntimeError(f"投稿状態の取得に失敗: HTTP {resp.status_code}")
    body = resp.json()
    state = json.loads(base64.b64decode(body["content"]).decode())
    return state, body["sha"]


def _already_posted(now):
    if os.environ.get("FORCE_POST") == "1":
        print("  [重複ガード] FORCE_POST=1 のため判定をスキップ")
        return False
    try:
        state, _ = _read_post_state()
    except Exception as e:
        # 読めないときは投稿する。重複より「予報が出ない」ほうが損失が大きい。
        print(f"  [警告] 投稿状態を読めませんでした（投稿を継続します）: {e}")
        return False
    if state is None:
        print("  [重複ガード] GITHUB_TOKEN 未設定のため判定不能（投稿を継続します）")
        return False
    return state.get("last_post_key") == _post_key(now)


def _mark_posted(now, post_id):
    """投稿成功を記録する。ここが失敗しても投稿自体は成功しているのでジョブは落とさない。"""
    try:
        _, sha = _read_post_state()
        payload = {
            "last_post_key": _post_key(now),
            "posted_at": now.strftime("%Y-%m-%d %H:%M:%S"),
            "post_id": post_id,
        }
        body = json.dumps(payload, ensure_ascii=False, indent=2)
        content = base64.b64encode((body + chr(10)).encode()).decode()
        data = {"message": f"Auto: last_post {_post_key(now)}",
                "content": content, "branch": "main"}
        if sha:
            data["sha"] = sha
        resp = requests.put(_post_state_api(), json=data,
                            headers=_post_state_headers(), timeout=15)
        if resp.status_code not in (200, 201):
            print(f"  [警告] 投稿状態の記録に失敗: HTTP {resp.status_code}")
        else:
            print(f"  [重複ガード] 投稿済みとして記録: {_post_key(now)}")
    except Exception as e:
        print(f"  [警告] 投稿状態の記録に失敗（投稿自体は成功）: {e}")


def _post_to_instagram(image_urls, caption):
    access_token = os.environ.get("INSTAGRAM_ACCESS_TOKEN")
    user_id      = os.environ.get("INSTAGRAM_USER_ID")

    if not access_token or not user_id:
        print("  [スキップ] INSTAGRAM_ACCESS_TOKEN / INSTAGRAM_USER_ID 未設定")
        return None
    if not image_urls:
        print("  [スキップ] 画像URLなし")
        return None

    try:
        # GitHub Pages が実際にファイルを配信するまでポーリング（最大5分）
        check_url = image_urls[0]
        print(f"  [Instagram] GitHub Pages 配信確認中（最大5分）: {check_url}")
        deadline = time.time() + 300
        while time.time() < deadline:
            try:
                r = requests.head(check_url, timeout=10, allow_redirects=True)
                if r.status_code == 200:
                    print(f"  [Instagram] 配信確認OK（{r.status_code}）→ Instagram投稿開始")
                    break
                print(f"  [Instagram] まだ未配信（{r.status_code}）... 15秒後再確認")
            except Exception:
                print("  [Instagram] 疎通確認エラー... 15秒後再確認")
            time.sleep(15)
        else:
            print("  [警告] GitHub Pages 5分待機タイムアウト。そのまま試行します")

        media_ids = []
        for img_url in image_urls:
            resp = requests.post(
                f"https://graph.instagram.com/v25.0/{user_id}/media",
                params={"image_url": img_url, "is_carousel_item": "true",
                        "access_token": access_token}
            )
            data = resp.json()
            if "id" not in data:
                raise InstagramPostError(f"メディアコンテナ作成失敗: {data}")
            media_ids.append(data["id"])

        resp = requests.post(
            f"https://graph.instagram.com/v25.0/{user_id}/media",
            params={"media_type": "CAROUSEL", "children": ",".join(media_ids),
                    "caption": caption, "access_token": access_token}
        )
        data = resp.json()
        if "id" not in data:
            raise InstagramPostError(f"カルーセル作成失敗: {data}")
        carousel_id = data["id"]

        print("  [Instagram] 処理待機（30秒）...")
        time.sleep(30)

        resp = requests.post(
            f"https://graph.instagram.com/v25.0/{user_id}/media_publish",
            params={"creation_id": carousel_id, "access_token": access_token}
        )
        data = resp.json()
        if "id" not in data:
            raise InstagramPostError(f"投稿の公開に失敗: {data}")

        print(f"  ✅ Instagram投稿完了: post_id={data['id']}")
        return data["id"]

    except InstagramPostError:
        raise
    except Exception as e:
        raise InstagramPostError(f"Instagram投稿エラー: {e}") from e


# ============================================================
# キャプション生成（座間味と同じ6段階コメント）
# ============================================================

def _build_caption(forecast, now, suspensions=None):
    short = forecast["short_term"]
    lt    = forecast["long_term"]
    s0    = short[0] if short else {}
    s1    = short[1] if len(short) > 1 else {}
    suspensions = suspensions or []

    # 長期期間の表記
    if lt["has_risk"]:
        lt_period_ja = lt["risk_period"]
        lt_period_en = lt["risk_period_en"]
    else:
        if lt.get("days"):
            d0 = datetime.strptime(lt["days"][0]["date"], "%Y-%m-%d")
            d1 = datetime.strptime(lt["days"][-1]["date"], "%Y-%m-%d")
            lt_period_ja = f"{d0.strftime('%-m/%-d')}〜{d1.strftime('%-m/%-d')}"
            lt_period_en = f"{d0.strftime('%b %-d')} - {d1.strftime('%b %-d')}"
        else:
            lt_period_ja = lt.get("lt_period_ja", "")
            lt_period_en = lt.get("lt_period_en", "")

    # リスクティア判定（短期最大値）
    max_pct = max(
        s0.get("hs_pct") or 0, s0.get("fe_pct") or 0,
        s1.get("hs_pct") or 0, s1.get("fe_pct") or 0,
    )

    if max_pct <= 10 and not lt["has_risk"]:
        comment_ja = (
            "\n🟢 今週は全日程で欠航リスク極めて低め！\n"
            "島滞在中の方も、これから渡航予定の方も安心してプランを組めそうです。\n"
        )
        comment_en = "\n🟢 Low cancellation risk all week — great time to visit!\n"
    elif max_pct <= 30 and not lt["has_risk"]:
        comment_ja = (
            "\n🟢 欠航リスクは低い見込みです。\n"
            "出発前に最新の予報をご確認ください。\n"
        )
        comment_en = "\n🟢 Cancellation risk looks low. Check the forecast before departure.\n"
    elif max_pct <= 30 and lt["has_risk"]:
        comment_ja = (
            "\n🟡 短期は問題なし。ただし来週以降に荒れる可能性があります。\n"
            "引き続き予報をチェックしていきましょう。\n"
        )
        comment_en = "\n🟡 Short-term looks fine, but rougher conditions may develop next week. Keep an eye on forecasts.\n"
    elif max_pct <= 60:
        comment_ja = (
            "\n🟡 現時点では運航見込みですが、想定より荒天が進めば欠航リスクが出てきます。\n"
            "最新情報をご確認ください。\n"
        )
        comment_en = "\n🟡 Currently operating, but cancellations may occur if conditions worsen. Check latest info.\n"
    elif max_pct <= 80:
        comment_ja = (
            "\n🔴 高速船の欠航リスクが高い状況です。\n"
            "旅程は余裕をもって組んでおくことをおすすめします。最新情報は渡嘉敷村HPへ。\n"
        )
        comment_en = "\n🔴 High cancellation risk. Consider scheduling with some flexibility. Check Tokashiki Village website.\n"
    else:
        comment_ja = (
            "\n🚨 欠航可能性が非常に高い状況です。\n"
            "島内滞在中の方は帰島日の前倒しをご検討ください。渡航予定の方は旅程変更も選択肢に。\n"
        )
        comment_en = "\n🚨 Very high cancellation risk. Guests on the island should consider an earlier return. Those planning to visit may want to reconsider.\n"

    hs0 = s0.get("hs_pct", "—"); fe0 = s0.get("fe_pct", "—")
    hs1 = s1.get("hs_pct", "—"); fe1 = s1.get("fe_pct", "—")
    dl0 = s0.get("date_label", ""); dl1 = s1.get("date_label", "")
    dl0_en = s0.get("date_label_en", ""); dl1_en = s1.get("date_label_en", "")

    # 計画運休ライン（期限内のものだけ表示）
    sus_lines_ja = "".join(
        f"⚠️ {s['vessel_ja']}は{s['start'][5:].replace('-','/')}〜"
        f"{s['end'][5:].replace('-','/')} {s['reason_ja']}運休中\n"
        for s in suspensions
    )
    sus_lines_en = "".join(
        f"⚠️ {s['vessel_en']} Suspended {s['start'][5:].replace('-','/')} - "
        f"{s['end'][5:].replace('-','/')} ({s['reason_en']})\n"
        for s in suspensions
    )

    ig_caption = (
        f"{forecast['update_date_ja']} {forecast['generated_at_label']}\n"
        f"渡嘉敷航路 欠航リスク予報\n"
        f"\n"
        + sus_lines_ja
        + f"■欠航可能性\n"
        f"明日 {dl0}  高速船 {hs0}% / フェリー {fe0}%\n"
        f"明後日 {dl1} 高速船 {hs1}% / フェリー {fe1}%\n"
        f"長期（{lt_period_ja}）: {lt['risk_period'] if lt['has_risk'] else '懸念なし'} "
        f"最大{lt['max_pct']}%\n"
        + comment_ja
        + "⚠️ AI予測・参考値です。公式情報は渡嘉敷村フェリー公式HPを参照ください。\n"
        + "#渡嘉敷島 #渡嘉敷 #慶良間諸島 #沖縄離島 #欠航予報\n"
        + "\n"
        + "\n"
        + f"{forecast['update_date_en']} updated\n"
        + "Tokashiki Route  Cancellation Risk Forecast\n"
        + "\n"
        + sus_lines_en
        + "■Boat/Ferry Cancellation Risk\n"
        + f"Tomorrow ({dl0_en}) Marine Liner {hs0}% / Ferry {fe0}%\n"
        + f"Day After ({dl1_en}) Marine Liner {hs1}% / Ferry {fe1}%\n"
        + f"Long-term ({lt_period_en}): "
        + f"{'No Significant Risk' if not lt['has_risk'] else lt['risk_period_en']}, "
        + f"max.{lt['max_pct']}%\n"
        + comment_en
        + "⚠️ AI-based estimate, for reference only\n"
        + "Check official Tokashiki Village website for confirmed info\n"
        + "\n"
        + "#KeramaIslands #Tokashiki #OkinawaFerry #JapanTravel"
    )
    return ig_caption


# ============================================================
# メイン
# ============================================================

def run_tokashiki_publisher(weather=None):
    """
    tokashiki_logger.py から呼び出すエントリーポイント。
    weather: ロガー取得済みの気象dict（省略時は内部で取得）
    """
    now = datetime.now(JST)
    print(f"\n{'='*50}")
    print(f"Tokashiki Publisher: {now.strftime('%Y-%m-%d %H:%M')}")
    print("="*50)

    # [P0] 計画運休情報読み込み（期限内のものだけ抽出）
    suspensions = _load_active_suspensions()

    # [P1] 8日間予報取得（3〜7日先 = index3〜7 を表示するため8日分必要）
    print("\n[P1] 8日間予報取得中（3地点 batched）...")
    day_list = _fetch_forecast(days=8)

    # Day1 はロガー取得済みデータで上書き（精度優先・API節約）
    if weather and len(day_list) > 1:
        d = day_list[1]
        d["max_wave"]  = weather.get("tmr_max_wave")  or d["max_wave"]
        d["max_swell"] = weather.get("tmr_max_swell") or d["max_swell"]
        d["max_wind"]  = weather.get("tmr_max_wind")  or d["max_wind"]

    # [P1b] JMA データ取得
    print("\n[P1b] JMAデータ取得中...")
    jma_waves = _get_jma_forecast_waves()
    jma_prob  = _get_jma_probability()
    print(f"  JMA波浪: {jma_waves}")
    print(f"  JMA確率: {jma_prob}")

    # [P1c] 構造化予報データ構築
    forecast = _build_forecast_data(day_list, jma_waves=jma_waves, jma_prob=jma_prob)

    # [P1d] Slack アラート（61%以上の場合のみ）
    _send_slack_alert(forecast, now)

    # [P2] 画像生成
    print("\n[P2] 画像生成中...")
    output_dir = "/tmp/tokashiki_images"
    os.makedirs(output_dir, exist_ok=True)
    ts    = now.strftime("%Y%m%d_%H%M")
    paths = [
        f"{output_dir}/tk_img1_short_{ts}.png",
        f"{output_dir}/tk_img2_longterm_{ts}.png",
        f"{output_dir}/tk_img3_weatherdata_{ts}.png",
    ]
    make_image_short(forecast,       paths[0])
    make_image_longterm(forecast,    paths[1])
    make_image_weatherdata(forecast, paths[2])

    # [P3] GitHub Pages アップロード（常に実行）
    print("\n[P3] GitHub Pages へ画像アップロード中...")
    image_urls = _upload_images_to_github(paths)

    # [P4] Instagram 投稿
    caption = _build_caption(forecast, now, suspensions=suspensions)

    # 午後便（12時以降）は欠航リスクが高い場合のみInstagram投稿（座間味と同じロジック）
    # 条件: 短期（明日・明後日）+ 長期（3〜7日先）全期間のいずれかで欠航確率 61% 以上
    is_afternoon_run = now.hour >= 12
    if is_afternoon_run:
        short = forecast.get("short_term", [])
        lt_days = forecast.get("long_term", {}).get("days", [])
        all_days = list(short[:2]) + list(lt_days)
        max_pct = max(
            (d.get(k) or 0)
            for d in all_days
            for k in ["hs_pct", "fe_pct"]
        ) if all_days else 0
        if max_pct < 61:
            print(f"  [午後便] 全期間最大欠航リスク {max_pct}% < 61% → Instagram投稿スキップ")
            print("\n✅ Tokashiki Publisher 完了")
            return
        print(f"  [午後便] 最大欠航リスク {max_pct}% ≥ 61% → Instagram投稿実行")

    print(f"\n[P4] Instagram 投稿中...")
    if _already_posted(now):
        print(f"  [スキップ] {_post_key(now)} はすでに投稿済み（重複起動）")
        print("\n✅ Tokashiki Publisher 完了")
        return

    try:
        post_id = _post_to_instagram(image_urls, caption)
    except Exception as e:
        # 握り潰すとトークン失効などで投稿が止まっても success のままになり気づけない。
        traceback.print_exc()
        _alert_slack(f"渡嘉敷: Instagram投稿に失敗しました: {e}")
        raise

    # スキップ時は post_id が None。記録すると fallback が塞がるので投稿できた時だけ記録する。
    if post_id:
        _mark_posted(now, post_id)

    print("\n✅ Tokashiki Publisher 完了")


if __name__ == "__main__":
    run_tokashiki_publisher()

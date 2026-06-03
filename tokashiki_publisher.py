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
import math
import time
import base64
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

def _calc_score(wave, swell, wind, has_warning=False):
    """0〜1 のスコアを返す。None は 0 扱い。"""
    wave_score    = min((wave  or 0) / 5.0,  1.0)
    swell_score   = min((swell or 0) / 4.0,  1.0)
    wind_score    = min((wind  or 0) / 20.0, 1.0)
    warning_score = 1.0 if has_warning else 0.0
    return (wave_score  * SCORE_WEIGHTS["wave"]    +
            swell_score * SCORE_WEIGHTS["swell"]   +
            wind_score  * SCORE_WEIGHTS["wind"]    +
            warning_score * SCORE_WEIGHTS["warning"])


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

def _fetch_forecast(days=8, timeout=30, max_retries=3):
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
        f"&hourly=wave_height,swell_wave_height"
        f"&timezone=Asia%2FTokyo&forecast_days={days}"
    )
    weather_url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lats}&longitude={lons}"
        f"&hourly=wind_speed_10m&wind_speed_unit=ms"
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

        def _worst(locs, key):
            best = None
            for loc in (locs or []):
                times_list = loc.get("hourly", {}).get("time", [])
                vals  = loc.get("hourly", {}).get(key, [])
                v = [v for t, v in zip(times_list, vals) if t.startswith(target) and v is not None]
                if v:
                    m = max(v)
                    best = m if best is None else max(best, m)
            return round(best, 2) if best is not None else None

        day_list.append({
            "date":      target,
            "max_wave":  _worst(marine_locs,  "wave_height"),
            "max_swell": _worst(marine_locs,  "swell_wave_height"),
            "max_wind":  _worst(weather_locs, "wind_speed_10m"),
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
        score = _calc_score(d.get("max_wave"), d.get("max_swell"), d.get("max_wind"))
        if d.get("max_wave") is not None:
            hs_pct = _score_to_pct_highspeed(score)
            fe_pct = _score_to_pct_ferry(score)
        else:
            hs_pct = fe_pct = None
        label_ja = "明日" if delta == 1 else "明後日"
        label_en = "Tomorrow" if delta == 1 else "Day After"
        short_term.append({
            "date":           d.get("date", dt.strftime("%Y-%m-%d")),
            "date_label":     dt.strftime("%-m/%-d"),
            "date_label_en":  dt.strftime("%b %-d"),
            "label_ja":       label_ja,
            "label_en":       label_en,
            "hs_pct":         hs_pct,
            "fe_pct":         fe_pct,
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
        score = _calc_score(d.get("max_wave"), d.get("max_swell"), d.get("max_wind"))
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
# 画像生成（座間味と同じレイアウト）
# ============================================================

def make_image_short(forecast, output_path):
    """
    画像①: 短期予報
    明日 / 明後日 × 高速船（マリンライナー）/ フェリーとかしき
    座間味の make_image_short と同じ2列構成。フッターに JMA データを表示。
    """
    short = forecast["short_term"]
    max_pct = max(
        (p for d in short for p in [d.get("hs_pct"), d.get("fe_pct")] if p is not None),
        default=10
    )
    img  = Image.new("RGB", IMG_SIZE, color=_hex_to_rgb(_get_bg_color(max_pct)))
    draw = ImageDraw.Draw(img)

    f = {
        "title_ja":  _load_font(FONT_BOLD,    44),
        "title_en":  _load_font(FONT_MEDIUM,  28),
        "date":      _load_font(FONT_MEDIUM,  34),
        "date_en":   _load_font(FONT_REGULAR, 24),
        "pct":       _load_font(FONT_BOLD,    86),
        "type_ja":   _load_font(FONT_MEDIUM,  28),
        "type_en":   _load_font(FONT_REGULAR, 20),
        "jma":       _load_font(FONT_REGULAR, 17),
        "xs":        _load_font(FONT_REGULAR, 17),
    }

    # タイトル
    draw.text((540, 44),  "渡嘉敷航路 欠航リスク予報",
              font=f["title_ja"], fill="white", anchor="mm")
    draw.text((540, 90),  "Tokashiki Route  /  Cancellation Risk Forecast",
              font=f["title_en"], fill=(255,255,255,210), anchor="mm")
    draw.line([(80, 116), (1000, 116)], fill=(255,255,255,100), width=1)

    # レイアウト定数
    HS_TOP, HS_BTM = 238, 492
    FE_TOP, FE_BTM = 498, 758
    DIVIDER_Y = 495

    positions = [270, 810]

    for i, day in enumerate(short[:2]):
        x   = positions[i]
        px1 = x - 222

        hs_pct = day.get("hs_pct")
        fe_pct = day.get("fe_pct")

        # 日付ヘッダー
        draw.text((x, 168), f"{day['label_ja']}  {day['date_label']}",
                  font=f["date"], fill="white", anchor="mm")
        draw.text((x, 208), f"{day['label_en']}  {day.get('date_label_en', '')}",
                  font=f["date_en"], fill=(255,255,255,180), anchor="mm")

        # 高速船セクション
        hs_mid = (HS_TOP + HS_BTM) // 2
        if hs_pct is not None:
            draw.text((x, hs_mid - 52), f"{hs_pct}%",
                      font=f["pct"], fill="white", anchor="mm")
        else:
            draw.text((x, hs_mid - 20), "—",
                      font=_load_font(FONT_BOLD, 60), fill=(200,200,200), anchor="mm")
        draw.text((x, hs_mid + 22), "マリンライナーとかしき",
                  font=f["type_ja"], fill=(255,255,255,220), anchor="mm")
        draw.text((x, hs_mid + 52), "Marine Liner Tokashiki",
                  font=f["type_en"], fill=(255,255,255,175), anchor="mm")

        # 区切り線
        draw.line([(px1 + 10, DIVIDER_Y), (x + 222 - 10, DIVIDER_Y)],
                  fill=(255,255,255,70), width=1)

        # フェリーセクション（JMA データを下部に表示）
        fe_mid = (FE_TOP + FE_BTM) // 2
        if fe_pct is not None:
            draw.text((x, fe_mid - 52), f"{fe_pct}%",
                      font=f["pct"], fill="white", anchor="mm")
        else:
            draw.text((x, fe_mid - 20), "—",
                      font=_load_font(FONT_BOLD, 60), fill=(200,200,200), anchor="mm")
        draw.text((x, fe_mid + 20), "フェリーとかしき",
                  font=f["type_ja"], fill=(255,255,255,220), anchor="mm")
        draw.text((x, fe_mid + 50), "Ferry Tokashiki",
                  font=f["type_en"], fill=(255,255,255,175), anchor="mm")
        # JMA データ（フェリーセクション下部）
        if day.get("jma_wave"):
            draw.text((x, fe_mid + 80),
                      f"気象庁: {day['jma_wave']}",
                      font=f["jma"], fill=(255,255,255,155), anchor="mm")
        if day.get("jma_prob"):
            draw.text((x, fe_mid + 100),
                      f"早期注意(波浪): {day['jma_prob']}",
                      font=f["jma"], fill=(255,255,255,155), anchor="mm")

    # 中央縦線・フッター
    draw.line([(540, 116), (540, 758)], fill=(255,255,255,55), width=1)
    draw.line([(80, 768), (1000, 768)], fill=(255,255,255,45), width=1)
    draw.text((540, 802), "※AI予測・参考値。運休は公式発表に基づきます。",
              font=f["xs"], fill=(255,255,255,155), anchor="mm")
    draw.text((540, 826), "*AI estimates for weather risk. Check official for cancellations.",
              font=f["xs"], fill=(255,255,255,125), anchor="mm")

    img.save(output_path)
    print(f"  画像①保存: {output_path}")


def make_image_longterm(forecast, output_path):
    """
    画像②: 長期予報（3〜7日先）
    座間味と同じレイアウト: リスク期間+最大% → 白棒グラフ2列
    """
    lt = forecast["long_term"]
    max_pct = lt["max_pct"]
    img  = Image.new("RGB", IMG_SIZE, color=_hex_to_rgb(_get_bg_color(max_pct)))
    draw = ImageDraw.Draw(img)

    f = {
        "title_ja": _load_font(FONT_BOLD,    44),
        "title_en": _load_font(FONT_MEDIUM,  26),
        "island":   _load_font(FONT_REGULAR, 22),
        "head":     _load_font(FONT_MEDIUM,  32),
        "head_en":  _load_font(FONT_REGULAR, 24),
        "period":   _load_font(FONT_BOLD,    64),
        "pct":      _load_font(FONT_BOLD,    76),
        "label":    _load_font(FONT_MEDIUM,  28),
        "label_en": _load_font(FONT_REGULAR, 22),
        "col_hd":   _load_font(FONT_MEDIUM,  22),
        "bar":      _load_font(FONT_REGULAR, 21),
        "badge":    _load_font(FONT_BOLD,    18),
        "xs":       _load_font(FONT_REGULAR, 17),
    }

    # タイトル
    draw.text((540, 46),  "フェリー欠航可能性 長期予報（3〜7日先）",
              font=f["title_ja"], fill="white", anchor="mm")
    draw.text((540, 86),  "Ferry Cancellation Risk  /  Long-term Forecast (3-7 days ahead)",
              font=f["title_en"], fill=(255,255,255,200), anchor="mm")
    draw.text((540, 112), "渡嘉敷島  Tokashiki Island",
              font=f["island"], fill=(255,255,255,160), anchor="mm")
    draw.line([(80, 128), (1000, 128)], fill=(255,255,255,100), width=1)

    if lt["has_risk"]:
        # リスク期間・最大%
        draw.text((540, 183), "欠航リスク期間  /  Risk Period",
                  font=f["head"], fill=(255,255,255,200), anchor="mm")
        draw.text((540, 255), lt["risk_period"],
                  font=f["period"], fill="white", anchor="mm")
        draw.text((540, 308), lt["risk_period_en"],
                  font=f["head_en"], fill=(255,255,255,180), anchor="mm")
        draw.line([(80, 328), (1000, 328)], fill=(255,255,255,70), width=1)

        # 高速船・フェリー最大%
        hs_max = max((d["hs_pct"] for d in lt["days"]), default=None)
        fe_max = max((d["fe_pct"] for d in lt["days"]), default=None)
        for x, pct, lja, len_ in [
            (270, hs_max, "高速船", "Marine Liner"),
            (810, fe_max, "フェリー", "Ferry"),
        ]:
            draw.text((x, 362), lja, font=f["label"], fill=(255,255,255,200), anchor="mm")
            draw.text((x, 388), len_, font=f["label_en"], fill=(255,255,255,170), anchor="mm")
            if pct is not None:
                draw.text((x, 453), f"{pct}%", font=f["pct"], fill="white", anchor="mm")
                draw.text((x, 500), "最大欠航可能性 / Max Risk",
                          font=f["label_en"], fill=(255,255,255,160), anchor="mm")
        draw.line([(540, 333), (540, 520)], fill=(255,255,255,60), width=1)
    else:
        draw.text((540, 300), "懸念なし  /  No Significant Risk",
                  font=f["period"], fill="white", anchor="mm")

    # 2列横棒グラフ
    FOOTER_LINE_Y = 960
    draw.line([(80, 530), (1000, 530)], fill=(255,255,255,70), width=1)
    draw.text((290, 552), "高速船  Marine Liner", font=f["col_hd"], fill="white", anchor="mm")
    draw.text((790, 552), "フェリー  Ferry",      font=f["col_hd"], fill="white", anchor="mm")

    bar_top, bar_h, row_sp = 580, 28, 72
    cols = [
        {"date_x": 155, "bar_x": 175, "bar_max": 270, "pct_x": 455, "key": "hs_pct"},
        {"date_x": 595, "bar_x": 615, "bar_max": 270, "pct_x": 895, "key": "fe_pct"},
    ]
    for i, d in enumerate(lt["days"][:5]):
        y     = bar_top + i * row_sp
        label = d["date_label"]
        for col in cols:
            pct   = d[col["key"]] or 0
            bar_w = int(col["bar_max"] * pct / 100)
            draw.text((col["date_x"], y + bar_h // 2), label,
                      font=f["bar"], fill="white", anchor="rm")
            draw.rectangle([(col["bar_x"], y), (col["bar_x"] + col["bar_max"], y + bar_h)],
                           fill=(0, 0, 0, 50))
            if bar_w > 0:
                draw.rectangle([(col["bar_x"], y), (col["bar_x"] + bar_w, y + bar_h)],
                               fill=(255, 255, 255, 210))
            draw.text((col["pct_x"], y + bar_h // 2), f"{pct}%",
                      font=f["bar"], fill="white", anchor="lm")

    draw.line([(540, 535), (540, FOOTER_LINE_Y)], fill=(255,255,255,50), width=1)
    draw.line([(80, FOOTER_LINE_Y), (1000, FOOTER_LINE_Y)], fill=(255,255,255,40), width=1)
    draw.text((540, 985), "※AI予測・参考値。公式情報は渡嘉敷村HPをご確認ください。",
              font=f["xs"], fill=(255,255,255,140), anchor="mm")
    draw.text((540, 1006), "*AI-based estimate. Check official Tokashiki Village website.",
              font=f["xs"], fill=(255,255,255,120), anchor="mm")

    img.save(output_path)
    print(f"  画像②保存: {output_path}")


def make_image_weatherdata(forecast, output_path):
    """
    画像③: 予報根拠データ（座間味と同じ3セクション構成）
    JMA / 数値予測 / 情報源
    """
    wd   = forecast.get("weather_data", {})
    img  = Image.new("RGB", IMG_SIZE, color=_hex_to_rgb("#0A1628"))
    draw = ImageDraw.Draw(img)

    f = {
        "title":    _load_font(FONT_BOLD,    40),
        "sec_hd":   _load_font(FONT_BOLD,    22),
        "label_ja": _load_font(FONT_REGULAR, 20),
        "label_en": _load_font(FONT_REGULAR, 17),
        "value":    _load_font(FONT_MEDIUM,  20),
        "src":      _load_font(FONT_REGULAR, 19),
        "foot":     _load_font(FONT_REGULAR, 17),
    }

    # タイトル
    draw.text((540, 68), "予報根拠データ  /  Forecast Data",
              font=f["title"], fill="white", anchor="mm")
    draw.line([(60, 100), (1020, 100)], fill="#334E7A", width=2)

    def section_header(y, ja, en):
        draw.rectangle([(60, y), (1020, y + 44)], fill="#1A3057")
        draw.text((80, y + 22), f"【{ja} / {en}】",
                  font=f["sec_hd"], fill="#7EB3F5", anchor="lm")
        return y + 60

    def row(y, icon, label_ja, label_en, value):
        draw.text((80,  y),     f"{icon} {label_ja}", font=f["label_ja"], fill="#BBDEFB", anchor="lm")
        draw.text((96,  y + 24), label_en,             font=f["label_en"], fill="#7986CB", anchor="lm")
        draw.text((1010, y + 12), str(value),           font=f["value"],    fill="white",   anchor="rm")
        return y + 58

    y = 112

    # 【気象庁 / JMA】
    y = section_header(y, "気象庁", "JMA")
    y = row(y, "🌊", "波高予報（明日）",           "Wave Height Forecast (Tomorrow)",  wd.get("jma_wave_tomorrow", "—"))
    y = row(y, "🌊", "波高予報（明後日）",          "Wave Height Forecast (Day After)", wd.get("jma_wave_dayafter", "—"))
    y = row(y, "⚠",  "早期注意情報・波浪（明日）",  "Early Warning Wave (Tomorrow)",    wd.get("jma_prob_tomorrow", "なし / None"))
    y = row(y, "⚠",  "早期注意情報・波浪（明後日）", "Early Warning Wave (Day After)",   wd.get("jma_prob_dayafter", "なし / None"))

    y += 12

    # 【数値予測 / Numerical Model】
    y = section_header(y, "数値予測", "Numerical Model")
    y = row(y, "📊", "明日 最大波高",   "Tomorrow Max Wave Height",  wd.get("num_max_wave",  "—"))
    y = row(y, "📊", "明日 最大うねり", "Tomorrow Max Swell Height", wd.get("num_max_swell", "—"))
    y = row(y, "💨", "明日 最大風速",   "Tomorrow Max Wind Speed",   wd.get("num_max_wind",  "—"))

    y += 12

    # 【情報源 / Sources】
    y = section_header(y, "情報源", "Sources")
    draw.text((80, y),      "気象庁（jma.go.jp）  /  Open-Meteo Marine API",
              font=f["src"], fill="#BBDEFB", anchor="lm")
    draw.text((80, y + 30), "渡嘉敷村HP（vill.tokashiki.okinawa.jp）",
              font=f["src"], fill="#BBDEFB", anchor="lm")
    draw.text((80, y + 60), "Tokashiki Village official site / JMA (jma.go.jp)",
              font=f["src"], fill="#7986CB", anchor="lm")

    # フッター
    draw.line([(60, 1020), (1020, 1020)], fill="#334E7A", width=1)
    draw.text((540, 1044), "※欠航判断は船会社・渡嘉敷村が行います。本データはAI予測の参考値です。",
              font=f["foot"], fill="#546E7A", anchor="mm")
    draw.text((540, 1066), "*Cancellation is determined by ferry operators. AI-based estimates for reference only.",
              font=f["foot"], fill="#455A64", anchor="mm")

    img.save(output_path)
    print(f"  画像③保存: {output_path}")


# ============================================================
# GitHub Pages アップロード・Instagram 投稿
# ============================================================

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


def _post_to_instagram(image_urls, caption):
    access_token = os.environ.get("INSTAGRAM_ACCESS_TOKEN")
    user_id      = os.environ.get("INSTAGRAM_USER_ID")

    if not access_token or not user_id:
        print("  [スキップ] INSTAGRAM_ACCESS_TOKEN / INSTAGRAM_USER_ID 未設定")
        return False
    if not image_urls:
        print("  [スキップ] 画像URLなし")
        return False

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
                f"https://graph.facebook.com/v25.0/{user_id}/media",
                params={"image_url": img_url, "is_carousel_item": "true",
                        "access_token": access_token}
            )
            data = resp.json()
            if "id" not in data:
                print(f"  [エラー] メディアコンテナ作成失敗: {data}")
                return False
            media_ids.append(data["id"])

        resp = requests.post(
            f"https://graph.facebook.com/v25.0/{user_id}/media",
            params={"media_type": "CAROUSEL", "children": ",".join(media_ids),
                    "caption": caption, "access_token": access_token}
        )
        data = resp.json()
        if "id" not in data:
            print(f"  [エラー] カルーセル作成失敗: {data}")
            return False
        carousel_id = data["id"]

        print("  [Instagram] 処理待機（30秒）...")
        time.sleep(30)

        resp = requests.post(
            f"https://graph.facebook.com/v25.0/{user_id}/media_publish",
            params={"creation_id": carousel_id, "access_token": access_token}
        )
        data = resp.json()
        if "id" not in data:
            print(f"  [エラー] 投稿失敗: {data}")
            return False

        print(f"  ✅ Instagram投稿完了: post_id={data['id']}")
        return True

    except Exception as e:
        print(f"  [警告] Instagram投稿エラー: {e}")
        return False


# ============================================================
# キャプション生成（座間味と同じ6段階コメント）
# ============================================================

def _build_caption(forecast, now):
    short = forecast["short_term"]
    lt    = forecast["long_term"]
    s0    = short[0] if short else {}
    s1    = short[1] if len(short) > 1 else {}

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

    ig_caption = (
        f"{forecast['update_date_ja']} {forecast['generated_at_label']}\n"
        f"渡嘉敷航路 欠航リスク予報\n"
        f"\n"
        f"■欠航可能性\n"
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
    caption = _build_caption(forecast, now)

    # 午後便（12時以降）は欠航リスクが高い場合のみInstagram投稿（座間味と同じロジック）
    # 条件: 明日 or 明後日の高速船 or フェリー欠航確率 61% 以上
    is_afternoon_run = now.hour >= 12
    if is_afternoon_run:
        short = forecast.get("short_term", [])
        max_pct = max(
            (d.get(k) or 0)
            for d in short[:2]
            for k in ["hs_pct", "fe_pct"]
        )
        if max_pct < 61:
            print(f"  [午後便] 欠航リスク最大 {max_pct}% < 61% → Instagram投稿スキップ")
            print("\n✅ Tokashiki Publisher 完了")
            return
        print(f"  [午後便] 欠航リスク最大 {max_pct}% ≥ 61% → Instagram投稿実行")

    print(f"\n[P4] Instagram 投稿中...")
    _post_to_instagram(image_urls, caption)

    print("\n✅ Tokashiki Publisher 完了")


if __name__ == "__main__":
    run_tokashiki_publisher()

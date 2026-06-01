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
  3枚目: 予報根拠データ
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


def _get_risk_text_color(pct):
    if pct is None or pct <= 30:  return "#66FF80"
    elif pct <= 60:               return "#FFD54F"
    elif pct <= 80:               return "#FF8A50"
    else:                         return "#FF6666"


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
# Open-Meteo: 複数地点 batched 取得（最悪値採用）
# ============================================================

def _fetch_forecast(days=7, timeout=30, max_retries=3):
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
                times = loc.get("hourly", {}).get("time", [])
                vals  = loc.get("hourly", {}).get(key, [])
                v = [v for t, v in zip(times, vals) if t.startswith(target) and v is not None]
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
# 予報データ構築
# ============================================================

def _build_forecast(day_list):
    """
    day_list から高速船・フェリーの欠航確率%を計算。
    戻り値: [{"date", "hs_pct", "fe_pct"}, ...]（7日分）
    """
    result = []
    for d in day_list:
        score   = _calc_score(d["max_wave"], d["max_swell"], d["max_wind"])
        hs_pct  = _score_to_pct_highspeed(score) if d["max_wave"] is not None else None
        fe_pct  = _score_to_pct_ferry(score)      if d["max_wave"] is not None else None
        result.append({
            "date":     d["date"],
            "hs_pct":   hs_pct,
            "fe_pct":   fe_pct,
            "max_wave": d["max_wave"],
            "max_swell":d["max_swell"],
            "max_wind": d["max_wind"],
        })
        if d["max_wave"] is not None:
            print(f"  {d['date']}: 波{d['max_wave']}m / 高速船{hs_pct}% / フェリー{fe_pct}%")
    return result


# ============================================================
# 画像生成
# ============================================================

def _fonts():
    return {
        "title":    _load_font(FONT_BOLD,    44),
        "title_en": _load_font(FONT_MEDIUM,  24),
        "head":     _load_font(FONT_BOLD,    32),
        "head_en":  _load_font(FONT_REGULAR, 22),
        "pct_big":  _load_font(FONT_BOLD,    86),
        "pct_med":  _load_font(FONT_BOLD,    60),
        "type_ja":  _load_font(FONT_MEDIUM,  28),
        "date":     _load_font(FONT_MEDIUM,  34),
        "date_en":  _load_font(FONT_REGULAR, 22),
        "bar":      _load_font(FONT_REGULAR, 22),
        "xs":       _load_font(FONT_REGULAR, 17),
        "sec":      _load_font(FONT_BOLD,    22),
        "val":      _load_font(FONT_MEDIUM,  20),
    }


def make_image_short(forecast, output_path):
    """
    画像①: 短期予報
    明日 / 明後日 × 高速船（マリンライナー）/ フェリーとかしき
    ferry-forecast の make_image_short と同じ2列構成。
    """
    now = datetime.now(JST)
    DAY_JA  = ["（月）","（火）","（水）","（木）","（金）","（土）","（日）"]
    DAY_EN  = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

    day1 = forecast[1] if len(forecast) > 1 else {}
    day2 = forecast[2] if len(forecast) > 2 else {}

    max_pct = max(
        (p for p in [day1.get("hs_pct"), day1.get("fe_pct"),
                     day2.get("hs_pct"), day2.get("fe_pct")] if p is not None),
        default=10
    )
    img  = Image.new("RGB", IMG_SIZE, color=_hex_to_rgb(_get_bg_color(max_pct)))
    draw = ImageDraw.Draw(img)
    f    = _fonts()

    # タイトル
    draw.text((540, 44),  "渡嘉敷航路 欠航リスク予報",
              font=f["title"], fill="white", anchor="mm")
    draw.text((540, 90),  "Tokashiki Route  /  Cancellation Risk Forecast",
              font=f["title_en"], fill=(255,255,255,210), anchor="mm")
    draw.line([(80, 116), (1000, 116)], fill=(255,255,255,80), width=1)

    # 2列（明日 / 明後日）
    DIVIDER_Y       = 495
    HS_TOP, HS_BTM  = 238, 492
    FE_TOP, FE_BTM  = 498, 758

    positions = [270, 810]
    days      = [day1, day2]

    for i, (d, x) in enumerate(zip(days, positions)):
        dt  = now + timedelta(days=i + 1)
        label_ja = "明日" if i == 0 else "明後日"
        label_en = "Tomorrow" if i == 0 else "Day After"
        date_ja  = f"{dt.month}/{dt.day}{DAY_JA[dt.weekday()]}"
        date_en  = f"{label_en}  {dt.strftime('%b')} {dt.day} ({DAY_EN[dt.weekday()]})"
        px1, px2 = x - 222, x + 222

        hs_pct = d.get("hs_pct")
        fe_pct = d.get("fe_pct")

        draw.text((x, 168), f"{label_ja}  {date_ja}",
                  font=f["date"], fill="white", anchor="mm")
        draw.text((x, 208), date_en,
                  font=f["date_en"], fill=(255,255,255,180), anchor="mm")

        # 高速船セクション
        hs_mid = (HS_TOP + HS_BTM) // 2
        if hs_pct is not None:
            draw.text((x, hs_mid - 52), f"{hs_pct}%",
                      font=f["pct_big"], fill="white", anchor="mm")
        else:
            draw.text((x, hs_mid - 20), "—",
                      font=f["pct_med"], fill=(200,200,200), anchor="mm")
        draw.text((x, hs_mid + 22), "マリンライナーとかしき",
                  font=f["type_ja"], fill=(255,255,255,220), anchor="mm")
        draw.text((x, hs_mid + 52), "Marine Liner Tokashiki",
                  font=_load_font(FONT_REGULAR, 20), fill=(255,255,255,175), anchor="mm")

        # 区切り線
        draw.line([(px1 + 10, DIVIDER_Y), (px2 - 10, DIVIDER_Y)],
                  fill=(255,255,255,70), width=1)

        # フェリーセクション
        fe_mid = (FE_TOP + FE_BTM) // 2
        if fe_pct is not None:
            draw.text((x, fe_mid - 46), f"{fe_pct}%",
                      font=f["pct_big"], fill="white", anchor="mm")
        else:
            draw.text((x, fe_mid - 20), "—",
                      font=f["pct_med"], fill=(200,200,200), anchor="mm")
        draw.text((x, fe_mid + 26), "フェリーとかしき",
                  font=f["type_ja"], fill=(255,255,255,220), anchor="mm")
        draw.text((x, fe_mid + 56), "Ferry Tokashiki",
                  font=_load_font(FONT_REGULAR, 20), fill=(255,255,255,175), anchor="mm")

    draw.line([(540, 116), (540, 758)], fill=(255,255,255,55), width=1)
    draw.line([(80, 768), (1000, 768)], fill=(255,255,255,45), width=1)
    draw.text((540, 802), "※AI予測・参考値。運休は公式発表に基づきます。",
              font=f["xs"], fill=(255,255,255,155), anchor="mm")
    draw.text((540, 826), "*AI estimates for weather risk. Check official for cancellations.",
              font=f["xs"], fill=(255,255,255,125), anchor="mm")

    img.save(output_path)
    print(f"  画像①保存: {output_path}")


def make_image_longterm(forecast, output_path):
    """画像②: 長期予報（3〜7日先・棒グラフ）"""
    now = datetime.now(JST)
    DAY_JA = ["月","火","水","木","金","土","日"]

    lt_days = []
    for delta in range(3, 8):
        d = forecast[delta] if delta < len(forecast) else {}
        lt_days.append({
            "date":  (now + timedelta(days=delta)).strftime("%Y-%m-%d"),
            "label": f"{(now + timedelta(days=delta)).month}/{(now + timedelta(days=delta)).day}"
                     f"({DAY_JA[(now + timedelta(days=delta)).weekday()]})",
            "hs_pct": d.get("hs_pct"),
            "fe_pct": d.get("fe_pct"),
        })

    max_pct = max(
        (p for d in lt_days for p in [d["hs_pct"], d["fe_pct"]] if p is not None),
        default=0
    )
    img  = Image.new("RGB", IMG_SIZE, color=_hex_to_rgb(_get_bg_color(max_pct)))
    draw = ImageDraw.Draw(img)
    f    = _fonts()

    draw.text((540, 44),  "長期予報（3〜7日先）",
              font=f["title"], fill="white", anchor="mm")
    draw.text((540, 88),  "Tokashiki Route  /  Long-term Risk Forecast  (3-7 days ahead)",
              font=f["title_en"], fill=(255,255,255,200), anchor="mm")
    draw.line([(60, 110), (1020, 110)], fill=(255,255,255,80), width=1)

    if max_pct >= 30:
        risk_days  = [d["label"] for d in lt_days
                      if (d["hs_pct"] or 0) >= 30 or (d["fe_pct"] or 0) >= 30]
        period_str = "  ".join(risk_days) if risk_days else "—"
        draw.text((540, 158), "注意が必要な期間  /  Risk Period",
                  font=f["head"], fill=(255,255,255,200), anchor="mm")
        draw.text((540, 228), period_str,
                  font=_load_font(FONT_BOLD, 26), fill="white", anchor="mm")
        draw.text((540, 284), f"最大欠航リスク  Max:  高速船 {max(d['hs_pct'] or 0 for d in lt_days)}%  "
                  f"フェリー {max(d['fe_pct'] or 0 for d in lt_days)}%",
                  font=f["head_en"], fill=(255,255,255,190), anchor="mm")
    else:
        draw.text((540, 220), "懸念なし  /  No Significant Risk",
                  font=f["head"], fill="white", anchor="mm")

    draw.line([(60, 318), (1020, 318)], fill=(255,255,255,60), width=1)

    # 2列棒グラフ（高速船 / フェリー）
    draw.text((290, 340), "高速船  Marine Liner", font=f["head_en"], fill="white", anchor="mm")
    draw.text((790, 340), "フェリー  Ferry",      font=f["head_en"], fill="white", anchor="mm")

    BAR_TOP = 375
    BAR_H   = 36
    ROW_SP  = 90
    cols    = [
        {"date_x": 80,  "bar_x": 100, "bar_max": 290, "pct_x": 400, "key": "hs_pct"},
        {"date_x": 545, "bar_x": 565, "bar_max": 290, "pct_x": 865, "key": "fe_pct"},
    ]

    for i, d in enumerate(lt_days):
        y = BAR_TOP + i * ROW_SP
        for col in cols:
            pct   = d[col["key"]] or 0
            bar_w = int(col["bar_max"] * pct / 100)
            draw.text((col["date_x"], y + BAR_H // 2), d["label"],
                      font=f["bar"], fill="white", anchor="rm")
            draw.rectangle([(col["bar_x"], y), (col["bar_x"] + col["bar_max"], y + BAR_H)],
                           fill=(0,0,0,55))
            if bar_w > 0:
                bc = tuple(min(255, int(c * 1.35)) for c in _hex_to_rgb(_get_bg_color(pct)))
                draw.rectangle([(col["bar_x"], y), (col["bar_x"] + bar_w, y + BAR_H)], fill=bc)
            pct_str = f"{pct}%" if d[col["key"]] is not None else "—"
            draw.text((col["pct_x"], y + BAR_H // 2), pct_str,
                      font=f["bar"], fill=_get_risk_text_color(pct), anchor="lm")

    draw.line([(60, 920), (1020, 920)], fill=(255,255,255,50), width=1)
    draw.text((540, 950), "※AI予測・参考値。欠航判断は公式発表に基づきます。",
              font=f["xs"], fill=(255,255,255,140), anchor="mm")
    draw.text((540, 972), "*AI estimates only. Check official announcements for cancellations.",
              font=f["xs"], fill=(255,255,255,110), anchor="mm")

    img.save(output_path)
    print(f"  画像②保存: {output_path}")


def make_image_weatherdata(forecast, output_path):
    """画像③: 根拠データ（明日・明後日の気象数値）"""
    now = datetime.now(JST)
    img  = Image.new("RGB", IMG_SIZE, color=_hex_to_rgb("#0A1628"))
    draw = ImageDraw.Draw(img)
    f    = _fonts()

    draw.text((540, 56),  "予報根拠データ  /  Forecast Data",
              font=f["title"], fill="white", anchor="mm")
    draw.line([(60, 90), (1020, 90)], fill="#334E7A", width=2)

    def section_header(y, ja, en):
        draw.rectangle([(60, y), (1020, y + 44)], fill="#1A3057")
        draw.text((80, y + 22), f"【{ja} / {en}】", font=f["sec"], fill="#7EB3F5", anchor="lm")
        return y + 60

    def row(y, label_ja, label_en, value):
        draw.text((80,  y),     label_ja,  font=_load_font(FONT_REGULAR, 20), fill="#BBDEFB", anchor="lm")
        draw.text((96,  y+24),  label_en,  font=_load_font(FONT_REGULAR, 17), fill="#7986CB", anchor="lm")
        draw.text((1010, y+12), str(value), font=_load_font(FONT_MEDIUM, 20),  fill="white",   anchor="rm")
        return y + 56

    def _fmt(v, unit=""): return f"{v:.1f}{unit}" if v is not None else "—"

    y = 104

    for delta in [1, 2]:
        d  = forecast[delta] if delta < len(forecast) else {}
        dt = now + timedelta(days=delta)
        DAY_JA = ["月","火","水","木","金","土","日"]
        label_ja = "明日" if delta == 1 else "明後日"
        label_en = "Tomorrow" if delta == 1 else "Day After"

        y = section_header(y, f"{label_ja} {dt.month}/{dt.day}({DAY_JA[dt.weekday()]})",
                           f"{label_en} {dt.strftime('%b %-d')}")
        y = row(y, "最大波高",  "Max Wave Height",         _fmt(d.get("max_wave"),  " m"))
        y = row(y, "最大うねり", "Max Swell Height",        _fmt(d.get("max_swell"), " m"))
        y = row(y, "最大風速",  "Max Wind Speed",           _fmt(d.get("max_wind"),  " m/s"))

        hs_pct = d.get("hs_pct")
        fe_pct = d.get("fe_pct")
        y = row(y, "高速船 欠航リスク", "Marine Liner Cancel Risk",
                f"{hs_pct}%" if hs_pct is not None else "—")
        y = row(y, "フェリー 欠航リスク", "Ferry Cancel Risk",
                f"{fe_pct}%" if fe_pct is not None else "—")
        y += 12

    y += 8
    draw.line([(60, y), (1020, y)], fill="#334E7A", width=1)
    draw.rectangle([(60, y + 8), (1020, y + 52)], fill="#1A3057")
    draw.text((80, y + 30), "【情報源】  Open-Meteo Marine API  /  渡嘉敷フェリーポータル tokashiki-ferry.jp",
              font=_load_font(FONT_REGULAR, 17), fill="#7EB3F5", anchor="lm")

    footer_y = y + 72
    draw.line([(60, footer_y), (1020, footer_y)], fill="#334E7A", width=1)
    draw.text((540, footer_y + 20), "※欠航判断は村営渡嘉敷村フェリー運航会社が行います。AI参考値。",
              font=_load_font(FONT_REGULAR, 16), fill="#546E7A", anchor="mm")
    draw.text((540, footer_y + 42), f"生成: {now.strftime('%Y-%m-%d %H:%M')} JST",
              font=_load_font(FONT_REGULAR, 16), fill="#37474F", anchor="mm")

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
        print("  [Instagram] GitHub Pages ビルド待機（90秒）...")
        time.sleep(90)

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


def _build_caption(forecast, now):
    tmr    = now + timedelta(days=1)
    DAY_JA = ["月","火","水","木","金","土","日"]
    DAY_EN = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    MON_EN = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    d1     = forecast[1] if len(forecast) > 1 else {}
    lines  = [
        f"🚢 渡嘉敷航路 欠航リスク予報  {tmr.month}/{tmr.day}({DAY_JA[tmr.weekday()]})",
        f"🚢 Tokashiki Route  Cancellation Risk  {MON_EN[tmr.month-1]} {tmr.day} ({DAY_EN[tmr.weekday()]})",
        "",
    ]
    hs_pct = d1.get("hs_pct")
    fe_pct = d1.get("fe_pct")
    if hs_pct is not None:
        icon = "🔴" if hs_pct >= 70 else ("🟡" if hs_pct >= 40 else "🟢")
        lines.append(f"{icon} マリンライナーとかしき / Marine Liner Tokashiki: {hs_pct}%")
    if fe_pct is not None:
        icon = "🔴" if fe_pct >= 70 else ("🟡" if fe_pct >= 40 else "🟢")
        lines.append(f"{icon} フェリーとかしき / Ferry Tokashiki: {fe_pct}%")
    lines += [
        "",
        "📊 詳細は画像スワイプでご確認ください。/ Swipe for details.",
        "⚠️ AI予測・参考値。欠航判断は公式HPをご確認ください。",
        "⚠️ AI estimates only. Check official site for cancellations.",
        "",
        "#渡嘉敷島 #渡嘉敷 #慶良間諸島 #沖縄離島 #欠航予報",
        "#Tokashiki #KeramaIslands #OkinawaFerry #JapanTravel",
    ]
    return "\n".join(lines)


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
    if weather:
        if len(day_list) > 1:
            d = day_list[1]
            d["max_wave"]  = weather.get("tmr_max_wave")  or d["max_wave"]
            d["max_swell"] = weather.get("tmr_max_swell") or d["max_swell"]
            d["max_wind"]  = weather.get("tmr_max_wind")  or d["max_wind"]

    forecast = _build_forecast(day_list)

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
    print(f"\n[P4] Instagram 投稿中...")
    _post_to_instagram(image_urls, caption)

    print("\n✅ Tokashiki Publisher 完了")


if __name__ == "__main__":
    run_tokashiki_publisher()

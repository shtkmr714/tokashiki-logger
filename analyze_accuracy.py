"""
analyze_accuracy.py
過去1ヶ月の運航実績 vs 予測確率を比較して精度を評価する。
GitHub Actions から一時的に実行するスクリプト（本番ロジックは変更しない）。

出力:
  - 座間味: daily_forecast vs daily_operation_log の比較
  - 渡嘉敷: tokashiki_operation_log の波高データから予測値を再計算して比較
"""

import os
import json
import math
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

JST = ZoneInfo("Asia/Tokyo")

# ============================================================
# 共通ユーティリティ
# ============================================================

def sigmoid(score, inflection, steepness):
    pct = 100 / (1 + math.exp(-steepness * (score - inflection)))
    return int(round(min(max(pct, 1), 99)))


def calc_score(wave, swell, wind):
    w = min((wave  or 0) / 5.0,  1.0) * 0.35
    s = min((swell or 0) / 4.0,  1.0) * 0.30
    v = min((wind  or 0) / 20.0, 1.0) * 0.20
    return round(w + s + v, 3)


def connect_sheets():
    import gspread
    from google.oauth2.service_account import Credentials
    svc = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not svc:
        raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_JSON not set")
    creds = Credentials.from_service_account_info(
        json.loads(svc),
        scopes=["https://spreadsheets.google.com/feeds",
                "https://www.googleapis.com/auth/drive"]
    )
    return gspread.authorize(creds)


def pct_label(pct):
    if pct is None: return "—"
    if pct >= 81:   return f"{pct}% 🔴"
    if pct >= 61:   return f"{pct}% 🟠"
    if pct >= 31:   return f"{pct}% 🟡"
    return f"{pct}% 🟢"


def confusion(pred_list, actual_list, threshold):
    """二値混同行列: pred>=threshold を「欠航予測」、actual==1 を「実際に欠航」"""
    tp = fp = tn = fn = 0
    for p, a in zip(pred_list, actual_list):
        if p is None or a is None:
            continue
        if p >= threshold and a == 1:   tp += 1
        elif p >= threshold and a == 0: fp += 1
        elif p < threshold  and a == 0: tn += 1
        elif p < threshold  and a == 1: fn += 1
    return tp, fp, tn, fn


def print_confusion(label, tp, fp, tn, fn):
    total = tp + fp + tn + fn
    if total == 0:
        print(f"  {label}: データなし")
        return
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    acc  = (tp + tn) / total
    print(f"  {label}: TP={tp} FP={fp} TN={tn} FN={fn} | "
          f"精度={prec:.0%} 検出率={rec:.0%} 正解率={acc:.0%}")


def calibration_buckets(pred_list, actual_list):
    """10%刻みのキャリブレーション: 予測確率 vs 実際の欠航率"""
    buckets = {i: {"n": 0, "cancel": 0} for i in range(0, 100, 10)}
    for p, a in zip(pred_list, actual_list):
        if p is None or a is None: continue
        b = min(int(p // 10) * 10, 90)
        buckets[b]["n"] += 1
        buckets[b]["cancel"] += a
    lines = []
    for b in sorted(buckets):
        n = buckets[b]["n"]
        if n == 0: continue
        actual_rate = buckets[b]["cancel"] / n
        lines.append(f"  予測{b:2d}〜{b+9}%: 実欠航率={actual_rate:.0%} (n={n})")
    return "\n".join(lines)


def brier_score(pred_list, actual_list):
    pairs = [(p, a) for p, a in zip(pred_list, actual_list) if p is not None and a is not None]
    if not pairs: return None
    return sum((p/100 - a)**2 for p, a in pairs) / len(pairs)


# ============================================================
# 座間味 分析
# ============================================================

def analyze_zamami(gc):
    print("\n" + "="*60)
    print("【座間味航路 精度分析】")
    print("="*60)

    sheets_id = os.environ.get("GOOGLE_SHEETS_ID")
    if not sheets_id:
        print("  [スキップ] GOOGLE_SHEETS_ID 未設定")
        return

    sh = gc.open_by_key(sheets_id)
    cutoff = (datetime.now(JST) - timedelta(days=31)).date()

    # daily_forecast（予測値）
    try:
        wf = sh.worksheet("daily_forecast")
        forecasts = {}  # target_date -> {hs_pct, fe_pct}
        for row in wf.get_all_records():
            d = row.get("target_date", "")
            if not d: continue
            try:
                if datetime.strptime(d, "%Y-%m-%d").date() < cutoff: continue
            except: continue
            hs = row.get("predicted_pct_highspeed")
            fe = row.get("predicted_pct_ferry")
            # 同日の最新（最後の）予測を使う
            if d not in forecasts:
                forecasts[d] = {"hs": None, "fe": None}
            if hs: forecasts[d]["hs"] = int(hs)
            if fe: forecasts[d]["fe"] = int(fe)
        print(f"\n  daily_forecast: {len(forecasts)}日分 (>= {cutoff})")
    except Exception as e:
        print(f"  [警告] daily_forecast 読み込みエラー: {e}")
        forecasts = {}

    # daily_operation_log（実績）
    try:
        wl = sh.worksheet("daily_operation_log")
        actuals = {}  # date -> {hs_cancel, fe_cancel}
        for row in wl.get_all_records():
            d = row.get("date", "")
            if not d: continue
            try:
                if datetime.strptime(d, "%Y-%m-%d").date() < cutoff: continue
            except: continue
            hs_reason = str(row.get("hs_cancel_reason", "none")).lower()
            fe_reason = str(row.get("ferry_cancel_reason", "none")).lower()
            hs_op1 = row.get("hs_bin1_operated", 1)
            hs_op2 = row.get("hs_bin2_operated", 1)
            fe_op  = row.get("ferry_operated", 1)
            # 気象欠航フラグ（weather の場合のみカウント）
            hs_wx = 1 if (("weather" in hs_reason) and (hs_op1 == 0 or hs_op2 == 0)) else 0
            fe_wx = 1 if (("weather" in fe_reason) and fe_op == 0) else 0
            actuals[d] = {
                "hs_cancel": hs_wx,
                "fe_cancel": fe_wx,
                "hs_reason": hs_reason,
                "fe_reason": fe_reason,
            }
        print(f"  daily_operation_log: {len(actuals)}日分")
    except Exception as e:
        print(f"  [警告] daily_operation_log 読み込みエラー: {e}")
        actuals = {}

    # 照合
    common = sorted(set(forecasts) & set(actuals))
    if not common:
        print("  [警告] 共通日付なし")
        return
    print(f"  照合可能日数: {len(common)}日 ({common[0]} 〜 {common[-1]})")

    hs_pred = [forecasts[d]["hs"] for d in common]
    fe_pred = [forecasts[d]["fe"] for d in common]
    hs_act  = [actuals[d]["hs_cancel"] for d in common]
    fe_act  = [actuals[d]["fe_cancel"] for d in common]

    # 欠航日数
    hs_cancel_days = sum(hs_act)
    fe_cancel_days = sum(fe_act)
    print(f"\n  気象欠航実績: 高速船={hs_cancel_days}日 / フェリー={fe_cancel_days}日 / {len(common)}日中")

    # キャリブレーション
    print("\n  [高速船 キャリブレーション（予測% vs 実欠航率）]")
    print(calibration_buckets(hs_pred, hs_act) or "  データなし")

    print("\n  [フェリー キャリブレーション]")
    print(calibration_buckets(fe_pred, fe_act) or "  データなし")

    # 混同行列（閾値 31%, 61%, 81%）
    print("\n  [高速船 混同行列]")
    for thr in [31, 61, 81]:
        print_confusion(f"閾値{thr}%", *confusion(hs_pred, hs_act, thr))

    print("\n  [フェリー 混同行列]")
    for thr in [31, 61, 81]:
        print_confusion(f"閾値{thr}%", *confusion(fe_pred, fe_act, thr))

    # Brier score
    bs_hs = brier_score(hs_pred, hs_act)
    bs_fe = brier_score(fe_pred, fe_act)
    if bs_hs: print(f"\n  Brier Score: 高速船={bs_hs:.3f}  フェリー={bs_fe:.3f}")
    print("  ※ Brier Score: 0=完璧、0.25=常に50%予測と同等、小さいほど良い")

    # 注目ケース：FN（見逃し）とFP（空振り）
    print("\n  [主な見逃し（実際に気象欠航したが予測が低かった日）]")
    fn_cases = [(d, forecasts[d]["hs"], actuals[d]) for d in common
                if (forecasts[d]["hs"] or 0) < 31 and actuals[d]["hs_cancel"] == 1]
    for d, p, a in fn_cases[:5]:
        print(f"    {d}: 予測高速船{p}% → 実際:{a['hs_reason']}")

    print("\n  [主な空振り（予測が高かったが実際は運航した日）]")
    fp_cases = [(d, forecasts[d]["hs"], actuals[d]) for d in common
                if (forecasts[d]["hs"] or 0) >= 61 and actuals[d]["hs_cancel"] == 0]
    for d, p, a in fp_cases[:5]:
        print(f"    {d}: 予測高速船{p}% → 実際:運航({a['hs_reason']})")


# ============================================================
# 渡嘉敷 分析
# ============================================================

def analyze_tokashiki(gc):
    print("\n" + "="*60)
    print("【渡嘉敷航路 精度分析】")
    print("="*60)

    sheets_id = os.environ.get("GOOGLE_SHEETS_ID_TOKASHIKI")
    if not sheets_id:
        print("  [スキップ] GOOGLE_SHEETS_ID_TOKASHIKI 未設定")
        return

    sh = gc.open_by_key(sheets_id)
    cutoff = (datetime.now(JST) - timedelta(days=31)).date()

    try:
        wl = sh.worksheet("tokashiki_operation_log")
        rows = wl.get_all_records()
    except Exception as e:
        print(f"  [エラー] シート読み込み失敗: {e}")
        return

    # 渡嘉敷は当日の波高データからリアルタイムに予測を再計算
    # （publisher の daily_forecast シートがないため）
    records = []
    for row in rows:
        d = row.get("date", "")
        if not d: continue
        try:
            if datetime.strptime(d, "%Y-%m-%d").date() < cutoff: continue
        except: continue

        wave = row.get("wave_max") or None
        swell = row.get("swell_max") or None
        wind = row.get("wind_max") or None
        if wave is None: continue

        try:
            wave  = float(wave)
            swell = float(swell) if swell else 0
            wind  = float(wind)  if wind  else 0
        except: continue

        score  = calc_score(wave, swell, wind)
        hs_pct = sigmoid(score, 0.42, 14.0)
        fe_pct = sigmoid(score, 0.52, 12.0)

        # 実績
        marine_reason = str(row.get("marine_cancel_reason", "none")).lower()
        ferry_reason  = str(row.get("ferry_cancel_reason",  "none")).lower()
        m_op1 = row.get("marine_bin1_operated", 1)
        m_op2 = row.get("marine_bin2_operated", 1)
        f_op1 = row.get("ferry_bin1_operated",  1)
        marine_wx = 1 if ("weather" in marine_reason) and (m_op1 == 0 or m_op2 == 0) else 0
        ferry_wx  = 1 if ("weather" in ferry_reason)  and f_op1 == 0 else 0

        records.append({
            "date": d, "wave": wave, "swell": swell, "wind": wind,
            "score": score, "hs_pct": hs_pct, "fe_pct": fe_pct,
            "marine_wx": marine_wx, "ferry_wx": ferry_wx,
            "marine_reason": marine_reason, "ferry_reason": ferry_reason,
        })

    if not records:
        print("  データなし（波高データがない日が多い可能性）")
        return

    print(f"\n  分析対象: {len(records)}日分 (>= {cutoff})")
    marine_cancel = sum(r["marine_wx"] for r in records)
    ferry_cancel  = sum(r["ferry_wx"]  for r in records)
    print(f"  気象欠航実績: マリンライナー={marine_cancel}日 / フェリー={ferry_cancel}日 / {len(records)}日中")

    hs_pred = [r["hs_pct"] for r in records]
    fe_pred = [r["fe_pct"] for r in records]
    hs_act  = [r["marine_wx"] for r in records]
    fe_act  = [r["ferry_wx"] for r in records]

    print("\n  [マリンライナー キャリブレーション（再計算予測% vs 実欠航率）]")
    print("  ※ 当日の実測波高からスコア式で予測を再計算しています")
    print(calibration_buckets(hs_pred, hs_act) or "  データなし")

    print("\n  [フェリー キャリブレーション]")
    print(calibration_buckets(fe_pred, fe_act) or "  データなし")

    print("\n  [マリンライナー 混同行列]")
    for thr in [31, 61, 81]:
        print_confusion(f"閾値{thr}%", *confusion(hs_pred, hs_act, thr))

    print("\n  [フェリー 混同行列]")
    for thr in [31, 61, 81]:
        print_confusion(f"閾値{thr}%", *confusion(fe_pred, fe_act, thr))

    bs_hs = brier_score(hs_pred, hs_act)
    bs_fe = brier_score(fe_pred, fe_act)
    if bs_hs: print(f"\n  Brier Score: マリンライナー={bs_hs:.3f}  フェリー={bs_fe:.3f}")

    # 波高別の欠航率（実測）
    print("\n  [波高別 実際の欠航率（渡嘉敷）]")
    wave_buckets = {}
    for r in records:
        b = int(r["wave"] // 0.5) * 5  # 0.5m刻み→0.5単位のbucket
        key = f"{r['wave']//0.5 * 0.5:.1f}m〜"
        if key not in wave_buckets:
            wave_buckets[key] = {"n": 0, "marine": 0, "ferry": 0}
        wave_buckets[key]["n"] += 1
        wave_buckets[key]["marine"] += r["marine_wx"]
        wave_buckets[key]["ferry"]  += r["ferry_wx"]
    for k in sorted(wave_buckets, key=lambda x: float(x.replace('m〜', ''))):
        v = wave_buckets[k]
        if v["n"] < 2: continue
        mr = v["marine"] / v["n"]
        fr = v["ferry"]  / v["n"]
        print(f"  波高{k} n={v['n']:2d} | マリン欠航率={mr:.0%}  フェリー欠航率={fr:.0%}")

    # 見逃し・空振りケース
    print("\n  [主な見逃し（マリンライナー）]")
    for r in records:
        if r["hs_pct"] < 31 and r["marine_wx"] == 1:
            print(f"    {r['date']}: 予測{r['hs_pct']}% 波{r['wave']}m → 欠航({r['marine_reason']})")

    print("\n  [主な空振り（マリンライナー, 予測61%以上で実際は運航）]")
    for r in records:
        if r["hs_pct"] >= 61 and r["marine_wx"] == 0:
            print(f"    {r['date']}: 予測{r['hs_pct']}% 波{r['wave']}m → 運航({r['marine_reason']})")


# ============================================================
# メイン
# ============================================================

if __name__ == "__main__":
    print(f"欠航予測 精度分析  実行時刻: {datetime.now(JST).strftime('%Y-%m-%d %H:%M')}")
    print(f"分析期間: 過去31日間")

    try:
        gc = connect_sheets()
        print("[Sheets] 接続成功")
    except Exception as e:
        print(f"[エラー] Sheets接続失敗: {e}")
        exit(1)

    analyze_zamami(gc)
    analyze_tokashiki(gc)

    print("\n\n" + "="*60)
    print("分析完了")
    print("="*60)

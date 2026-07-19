"""画像レイアウトの軽量回帰検査（CI用・渡嘉敷）。

2026-07 に八重山で発生した「カード/枠から文字・白背景がはみ出す」系の再発防止。
本番と同じ描画関数を worst-case データ（運休あり・100%・リスク1日・懸念なし）で呼び、
例外なく 1254² が出るか（クラッシュ・サイズ崩れ検出）を確認する。
渡嘉敷は船種2枠（高速船/フェリー）でルート名ピルが無く八重山ほど溢れないが、
テンプレ/座標変更時の回帰を push 時に検出する。

実行: python check_image_layout.py （失敗時 exit 1）
"""
import sys, os, tempfile
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from PIL import Image
import tokashiki_publisher as TP

FAILS = []
def check(cond, msg):
    print(("  OK  " if cond else " FAIL ") + msg)
    if not cond:
        FAILS.append(msg)

def day_short(lj, dl, hs_am, hs_pm, fe, sus_hs=False, sus_fe=False):
    return {"label_ja": lj, "date_label": dl, "label_en": "TOMORROW",
            "hs_am_pct": hs_am, "hs_pm_pct": hs_pm, "hs_pct": max(hs_am, hs_pm),
            "fe_pct": fe, "suspended_highspeed": sus_hs, "suspended_ferry": sus_fe}

def day_long(date, hs, fe, sus_hs=False, sus_fe=False):
    return {"date": date, "hs_pct": hs, "fe_pct": fe,
            "suspended_highspeed": sus_hs, "suspended_ferry": sus_fe}

def make_forecast(has_risk=True):
    lt = {"has_risk": has_risk, "max_pct": 100 if has_risk else 5,
          "days": [day_long("2026-07-22", 8, 1),
                   day_long("2026-07-23", 100, 1, sus_hs=True),
                   day_long("2026-07-24", 3, 1),
                   day_long("2026-07-25", 5, 1),
                   day_long("2026-07-26", 100, 100)]}
    if has_risk:
        lt["risk_period"] = "7/26"          # 単日
        lt["risk_period_en"] = "Jul 26"
    else:
        lt["risk_period"] = "懸念なし"
        lt["risk_period_en"] = "No concern"
    return {
        "short_term": [day_short("明日", "7/20", 100, 100, 100, sus_fe=True),
                       day_short("明後日", "7/21", 5, 5, 1)],
        "long_term": lt,
    }

tmp = tempfile.mkdtemp()
for tag, has_risk in (("risk-single-day+suspended", True), ("no-risk(懸念なし)", False)):
    fc = make_forecast(has_risk)
    for kind, fn in (("short", TP.make_image_short), ("long", TP.make_image_longterm)):
        p = os.path.join(tmp, f"tokashiki_{kind}_{has_risk}.png")
        try:
            fn(fc, p)
            im = Image.open(p)
            check(im.size == (1254, 1254),
                  f"tokashiki {kind} [{tag}] renders 1254x1254 (got {im.size})")
        except Exception as e:
            check(False, f"tokashiki {kind} [{tag}] raised: {e!r}")

print()
if FAILS:
    print(f"LAYOUT CHECK FAILED ({len(FAILS)} issue(s)):")
    for m in FAILS:
        print("  - " + m)
    sys.exit(1)
print("LAYOUT CHECK PASSED")

"""
check_columns.py
指定シートの各列について、直近データの「空欄率」を出して取得漏れを検出する汎用チェッカー。
環境変数 SHEET_ID（スプレッドシートID）と SHEET_TAB（タブ名）で対象を指定。
"""
import os, json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

JST = ZoneInfo("Asia/Tokyo")

def connect():
    import gspread
    from google.oauth2.service_account import Credentials
    creds = Credentials.from_service_account_info(
        json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]),
        scopes=["https://spreadsheets.google.com/feeds",
                "https://www.googleapis.com/auth/drive"])
    return gspread.authorize(creds)

if __name__ == "__main__":
    sheet_id = os.environ["SHEET_ID"]
    tab      = os.environ["SHEET_TAB"]
    gc = connect()
    ws = gc.open_by_key(sheet_id).worksheet(tab)
    rows = ws.get_all_records()
    print(f"=== {tab} 総行数 {len(rows)} ===")
    if not rows:
        print("データなし"); exit()

    # 直近30行 / 全体 の2軸で空欄率
    cutoff = (datetime.now(JST) - timedelta(days=31)).date()
    recent = []
    for r in rows:
        d = str(r.get("date","")).strip()
        try:
            if datetime.strptime(d, "%Y-%m-%d").date() >= cutoff:
                recent.append(r)
        except: pass

    cols = list(rows[0].keys())
    def empty_rate(records, col):
        if not records: return None
        empty = sum(1 for r in records if str(r.get(col,"")).strip() in ("", "None"))
        return empty / len(records)

    print(f"\n直近31日: {len(recent)}行 / 全体: {len(rows)}行\n")
    print(f"{'列名':<28} {'全体空欄率':>10} {'直近空欄率':>10}")
    print("-" * 52)
    for c in cols:
        all_r = empty_rate(rows, c)
        rec_r = empty_rate(recent, c)
        flag = ""
        if rec_r is not None and rec_r >= 0.9:
            flag = "  ⚠️ ほぼ空"
        elif rec_r is not None and rec_r >= 0.5:
            flag = "  △ 半数空"
        print(f"{c:<28} {all_r:>9.0%} {(rec_r if rec_r is not None else 0):>9.0%}{flag}")

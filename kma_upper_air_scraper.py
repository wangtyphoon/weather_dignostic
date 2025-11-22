import datetime as dt
import time
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json

BASE_TEMPLATE = "https://www.weather.go.kr/w/repositary/image/cht/img/{prefix}_{level_code}_{product}_{region}_{timestamp}.gif"
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.kma.go.kr/neng/image/chart/analysis-chart.do",
}
LEVELS_MAP = {
    "925": "up92", "850": "up85", "700": "up70", "500": "up50",
    "300": "up30", "200": "up20", "100": "up10",
}


def iter_hours(start: dt.datetime, end: dt.datetime, step_hours: int = 6):
    cur = start
    delta = dt.timedelta(hours=step_hours)
    while cur <= end:
        yield cur
        cur += delta


def build_url(prefix: str, level_code: str, product: str, region: str, t: dt.datetime) -> str:
    ts = t.strftime("%Y%m%d%H")
    return BASE_TEMPLATE.format(prefix=prefix, level_code=level_code, product=product, region=region, timestamp=ts)


def safe_filename(prefix: str, level_code: str, product: str, region: str, t: dt.datetime) -> str:
    return f"{prefix}_{level_code}_{product}_{region}_{t.strftime('%Y%m%d%H')}.gif"


def download_one(session: requests.Session, url: str, out_path: Path, retries: int = 3, backoff: float = 1.6):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    err_msg = ""
    for attempt in range(1, retries + 1):
        try:
            with session.get(url, stream=True, timeout=20) as r:
                if r.status_code == 200 and r.headers.get("Content-Type", "").startswith("image/"):
                    with open(out_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=64 * 1024):
                            if chunk:
                                f.write(chunk)
                    return True, ""
                else:
                    err_msg = f"HTTP {r.status_code} {r.headers.get('Content-Type','')}"
        except requests.RequestException as e:
            err_msg = str(e)
        if attempt < retries:
            time.sleep(backoff ** attempt)
    return False, err_msg


def align_to_cycle(t: dt.datetime, step_hours: int = 6) -> dt.datetime:
    """對齊到最近一個不晚於 t 的 step_hours 週期（例如 00/06/12/18）。"""
    t = t.replace(minute=0, second=0, microsecond=0)
    return t - dt.timedelta(hours=(t.hour % step_hours))

def run_kma_scraper(config: dict | str):
    """
    Run KMA upper-air chart scraper.
    config: dict 或 JSON 檔路徑 (str/Path)
    """

    # 如果傳的是路徑，就讀 JSON
    if isinstance(config, (str, Path)):
        config = json.loads(Path(config).read_text(encoding="utf-8"))

    # 預設值
    cfg = {
        "out": "kma_upper",
        "levels": ["925", "850", "700", "500", "300", "200", "100"],
        "hour_step": 6,
        "concurrency": 8,
        "prefix": "kim",
        "product": "anlmod",
        "region": "pa4",
        # 新增：若未指定，預設自動抓「最近 5 天」
        "auto_last_days": 5,
    }
    cfg.update(config)

    # ===== 時間範圍 =====
    step = int(cfg["hour_step"])

    if "latest_hours" in cfg:
        # 若使用者明確給了 latest_hours，仍然尊重這設定，但把起訖都對齊 6 小時
        now_utc = dt.datetime.utcnow()
        end = align_to_cycle(now_utc, step)
        start = align_to_cycle(end - dt.timedelta(hours=int(cfg["latest_hours"])), step)

    elif "start" in cfg and "end" in cfg:
        # 若仍然傳了手動起訖，也會幫你對齊
        start = align_to_cycle(dt.datetime.fromisoformat(cfg["start"]), step)
        end   = align_to_cycle(dt.datetime.fromisoformat(cfg["end"]),   step)

    else:
        # 預設：自動以「現在 UTC」往回抓滿 auto_last_days 天
        now_utc = dt.datetime.utcnow()
        end = align_to_cycle(now_utc, step)  # 最近的 00/06/12/18
        start = align_to_cycle(end - dt.timedelta(days=int(cfg.get("auto_last_days", 5))), step)

    # 驗證 level
    for lev in cfg["levels"]:
        if lev not in LEVELS_MAP:
            raise ValueError(f"Unsupported level {lev}")

    tasks = [(t, lev, LEVELS_MAP[lev]) for t in iter_hours(start, end, step) for lev in cfg["levels"]]

    out_dir = Path(cfg["out"])
    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)

    print(f"Planned: {len(tasks)} files → {out_dir.resolve()}")
    print(f"Time window (UTC): {start:%Y-%m-%d %H} → {end:%Y-%m-%d %H} (step={step}h)")
    success = fail = 0

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=cfg["concurrency"]) as ex:
        futures = []
        for t, lev_str, lev_code in tasks:
            url = build_url(cfg["prefix"], lev_code, cfg["product"], cfg["region"], t)
            out_path = out_dir / lev_str / safe_filename(cfg["prefix"], lev_code, cfg["product"], cfg["region"], t)
            futures.append(ex.submit(download_one, session, url, out_path))

        for (t, lev_str, lev_code), fut in zip(tasks, futures):
            ok, emsg = fut.result()
            tag = f"{t.strftime('%Y-%m-%d %H')} {lev_str}hPa"
            if ok:
                success += 1
                print(f"[OK]   {tag}")
            else:
                fail += 1
                print(f"[MISS] {tag} -> {emsg}")

    print(f"Done. Success={success}, Fail={fail}")


run_kma_scraper({
    "out": "kma_upper",
    "levels": ["850", "500", "200"],
    "hour_step": 6,
    # 不需要 start/end；可選擇性調整往回天數：
    "auto_last_days": 3,
})
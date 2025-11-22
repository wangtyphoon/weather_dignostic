"""Crawler for time-based imagery defined in download_document.txt."""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import requests

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None

TZ = ZoneInfo("Asia/Taipei") if ZoneInfo else dt.timezone(dt.timedelta(hours=8))

SESSION_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
    "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
    "Referer": "https://www.cwa.gov.tw/",
}

TIMESTAMP_REGEXES: Sequence[re.Pattern[str]] = (
    re.compile(r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}"),
    re.compile(r"\d{4}-\d{4}-\d{4}"),
    re.compile(r"\d{4}-\d{2}-\d{2}_\d{4}"),
    re.compile(r"\d{14}"),
    re.compile(r"\d{12}"),
    re.compile(r"\d{10}"),
    re.compile(r"\d{8}"),
)

FORMAT_BY_LENGTH: Dict[int, Sequence[str]] = {
    16: ("%Y-%m-%d-%H-%M",),
    15: ("%Y-%m-%d_%H%M",),
    14: ("%Y%m%d%H%M%S", "%Y-%m%d-%H%M"),
    12: ("%Y%m%d%H%M",),
    10: ("%Y%m%d%H", "%d%m%y%H%M"),
    8: ("%y%m%d%H", "%Y%m%d", "%d%m%y%H"),
}

FALLBACK_FORMATS: Sequence[str] = (
    "%Y%m%d%H%M%S",
    "%Y%m%d%H%M",
    "%Y%m%d%H",
    "%Y%m%d",
    "%d%m%y%H%M",
    "%d%m%y%H",
)

DEFAULT_TIMEOUT: Tuple[int, int] = (10, 30)


def sanitize(text: str) -> str:
    cleaned = re.sub(r"[<>:\"/\\|?*]", "_", text.strip())
    return cleaned or "untitled"


@dataclass
class TimestampPattern:
    token: str
    fmt: Optional[str]
    start: int
    end: int

    def render(self, ts: dt.datetime) -> str:
        if not self.fmt:
            raise ValueError("missing datetime format")
        return ts.strftime(self.fmt)

    def replace(self, url: str, replacement: str) -> str:
        return f"{url[:self.start]}{replacement}{url[self.end:]}"


def guess_datetime_format(token: str) -> Optional[str]:
    for fmt in FORMAT_BY_LENGTH.get(len(token), ()):  # exact-length candidates first
        try:
            dt.datetime.strptime(token, fmt)
            return fmt
        except ValueError:
            continue
    for fmt in FALLBACK_FORMATS:
        try:
            dt.datetime.strptime(token, fmt)
            return fmt
        except ValueError:
            continue
    return None


def extract_timestamp_pattern(url: str) -> Optional[TimestampPattern]:
    for regex in TIMESTAMP_REGEXES:
        match = regex.search(url)
        if match:
            token = match.group(0)
            fmt = guess_datetime_format(token)
            return TimestampPattern(token=token, fmt=fmt, start=match.start(), end=match.end())
    return None


@dataclass
class Dataset:
    category: str
    label: str
    url: str
    pattern: Optional[TimestampPattern] = None

    def __post_init__(self) -> None:
        if self.pattern is None:
            self.pattern = extract_timestamp_pattern(self.url)
        if self.pattern and 'skw___' in self.url.lower():
            self.pattern.fmt = '%y%m%d%H'

    def output_dir(self, root: Path) -> Path:
        return root / sanitize(self.category) / sanitize(self.label)


def existing_tokens(out_dir: Path) -> Set[str]:
    tokens: Set[str] = set()
    if not out_dir.exists():
        return tokens
    for path in out_dir.iterdir():
        if not path.is_file() or path.name.startswith("_"):
            continue
        tokens.add(path.name.split("_", 1)[0])
    return tokens


def align_time(ts: dt.datetime, interval: dt.timedelta) -> dt.datetime:
    seconds = max(int(interval.total_seconds()), 1)
    aligned_epoch = int(ts.timestamp()) - (int(ts.timestamp()) % seconds)
    return dt.datetime.fromtimestamp(aligned_epoch, tz=ts.tzinfo)


def determine_interval(dataset: Dataset, pattern: TimestampPattern) -> dt.timedelta:
    url_lower = dataset.url.lower()
    if "radar_rain" in url_lower:
        return dt.timedelta(minutes=5)
    if "radar" in url_lower:
        return dt.timedelta(minutes=10)
    if "satellite" in url_lower:
        return dt.timedelta(minutes=10)
    if "temperature" in url_lower:
        return dt.timedelta(minutes=10)
    if "analysis" in url_lower:
        return dt.timedelta(hours=6)
    if "fcst_img" in url_lower or "sfccombo" in url_lower:
        return dt.timedelta(hours=6)
    if "rainfall" in url_lower:
        if "qzj" in url_lower:
            return dt.timedelta(days=1)
        if "qzt" in url_lower:
            return dt.timedelta(hours=1)
    if "skw" in url_lower or "irisme" in url_lower:
        return dt.timedelta(hours=12)
    fmt = pattern.fmt or ""
    if "%S" in fmt:
        return dt.timedelta(minutes=1)
    if "%M" in fmt:
        return dt.timedelta(minutes=10)
    if "%H" in fmt:
        return dt.timedelta(hours=1)
    if "%d" in fmt:
        return dt.timedelta(days=1)
    return dt.timedelta(minutes=30)


@dataclass
class CrawlOptions:
    fast_days: float
    hourly_days: float
    daily_days: float
    slow_days: float
    max_failures: int
    since: Optional[dt.datetime]
    output_root: Path
    dry_run: bool


def compute_lookback(dataset: Dataset, interval: dt.timedelta, options: CrawlOptions) -> dt.timedelta:
    url_lower = dataset.url.lower()
    if "radar_rain" in url_lower:
        return dt.timedelta(days=min(options.fast_days, 1.0))
    seconds = interval.total_seconds()
    if seconds <= 1800:  # up to 30 minutes
        return dt.timedelta(days=options.fast_days)
    if seconds <= 3 * 3600:  # up to 3 hours
        return dt.timedelta(days=options.hourly_days)
    if seconds <= 48 * 3600:  # up to 2 days
        return dt.timedelta(days=options.daily_days)
    return dt.timedelta(days=options.slow_days)


def reference_now(dataset: Dataset) -> dt.datetime:
    url_lower = dataset.url.lower()
    if "irisme" in url_lower or "skw___" in url_lower:
        return dt.datetime.now(dt.timezone.utc)
    return dt.datetime.now(TZ)


def download_image(
    session: requests.Session,
    url: str,
    out_dir: Path,
    token: str,
    dry_run: bool,
) -> Tuple[str, str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(url).name or "download"
    needs_prefix = bool(token) and not filename.startswith(token)
    dest = out_dir / (f"{token}_{filename}" if needs_prefix else filename)
    if dest.exists() and not dry_run:
        return "skip", f"exists {dest.name}", dest
    if dry_run:
        return "ok", f"would download {dest.name}", dest
    try:
        with session.get(url, stream=True, timeout=DEFAULT_TIMEOUT) as resp:
            if resp.status_code != 200:
                return "fail", f"HTTP {resp.status_code}", dest
            content_type = resp.headers.get("Content-Type", "")
            if "image" not in content_type.lower():
                return "fail", f"unexpected content: {content_type}", dest
            with dest.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        fh.write(chunk)
    except requests.RequestException as exc:
        return "fail", f"request error: {exc}", dest
    return "ok", f"downloaded {dest.name}", dest


def parse_document(path: Path) -> List[Dataset]:
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    category: Optional[str] = None
    pending_label: Optional[str] = None
    datasets: List[Dataset] = []
    for line in lines:
        if not line:
            continue
        if line.startswith("http"):
            if not category:
                continue
            url = line
            if "radar_rain" in url.lower():
                pending_label = None
                continue
            label_source = url.split('?', 1)[0]
            lower_source = label_source.lower()
            if pending_label:
                label = pending_label
            else:
                label = Path(label_source).stem
                if "temperature" in lower_source:
                    dot = label.find(".")
                    if dot != -1 and dot + 1 < len(label):
                        label = label[dot + 1 :]
                    else:
                        label = "temperature"
                elif "analysis" in lower_source:
                    match = re.search(r"\d{8}", label)
                    if match:
                        cleaned = f"{label[:match.start()]}{label[match.end():]}"
                        cleaned = re.sub(r"_+", "_", cleaned)
                        label = cleaned.strip('_') or "surface-analysis-bw"
                    else:
                        label = "surface-analysis-bw"
            datasets.append(Dataset(category, label, url))
            pending_label = None
            continue
        if line[0].isdigit() and "." in line:
            dot = line.find(".")
            pending_label = line[dot + 1 :].strip()
            continue
        category = line
        pending_label = None
    return datasets


def create_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(SESSION_HEADERS)
    return session


def crawl_dataset(session: requests.Session, dataset: Dataset, options: CrawlOptions) -> Tuple[str, str]:
    out_dir = dataset.output_dir(options.output_root)
    pattern = dataset.pattern
    if not pattern or not pattern.fmt:
        status, message, _ = download_image(session, dataset.url, out_dir, "", options.dry_run)
        return status, message

    interval = determine_interval(dataset, pattern)
    lookback = compute_lookback(dataset, interval, options)
    seconds = max(int(interval.total_seconds()), 1)
    max_steps = max(int(lookback.total_seconds() // seconds) + 1, 1)
    anchor_base = reference_now(dataset)
    anchor = align_time(anchor_base, interval)
    url_lower = dataset.url.lower()
    if "fcst_img" in url_lower or "sfccombo" in url_lower:
        local_base = anchor_base.astimezone(TZ)
        for hour_mark in (18, 12, 6, 0):
            if local_base.hour >= hour_mark:
                anchor_local = local_base.replace(hour=hour_mark, minute=0, second=0, microsecond=0)
                break
        else:
            anchor_local = (local_base - dt.timedelta(days=1)).replace(hour=18, minute=0, second=0, microsecond=0)
        anchor = anchor_local
    elif "qzj" in url_lower:
        anchor = anchor_base.replace(hour=0, minute=0, second=0, microsecond=0)
        anchor -= interval
    tokens = existing_tokens(out_dir)
    consecutive_failures = 0
    downloaded = 0
    last_error: Optional[str] = None
    last_token: Optional[str] = None

    for step in range(max_steps):
        candidate_time = anchor - step * interval
        if options.since and candidate_time < options.since:
            break
        token_value = pattern.render(candidate_time)
        if token_value in tokens:
            continue
        candidate_url = pattern.replace(dataset.url, token_value)
        # print(f"Trying {candidate_url} ...")
        status, message, _ = download_image(session, candidate_url, out_dir, token_value, options.dry_run)
        if status == "ok":
            downloaded += 1
            tokens.add(token_value)
            consecutive_failures = 0
            last_token = token_value
        elif status == "skip":
            tokens.add(token_value)
            consecutive_failures = 0
        else:
            print(candidate_url)
            last_error = message
            consecutive_failures += 1
            if consecutive_failures >= options.max_failures:
                break

    if downloaded:
        tail = f" (latest token {last_token})" if last_token else ""
        note = "dry-run" if options.dry_run else "downloaded"
        return "ok", f"{note} {downloaded} files{tail}"
    if tokens:
        return "skip", "no new files"
    return "fail", last_error or "no images reachable"


def run(document_path: Path, options: CrawlOptions) -> None:
    datasets = parse_document(document_path)
    if not datasets:
        print(f"No entries found in {document_path}.")
        return

    session = create_session()
    stats = {"ok": 0, "skip": 0, "fail": 0}
    for dataset in datasets:
        status, message = crawl_dataset(session, dataset, options)
        stats[status] = stats.get(status, 0) + 1
        print(f"[{dataset.category} - {dataset.label}] {message}")
    print(
        "Summary -> ok: {ok}, skip: {skip}, fail: {fail}".format(
            ok=stats.get("ok", 0),
            skip=stats.get("skip", 0),
            fail=stats.get("fail", 0),
        )
    )


def main() -> None:
    document_path = Path("document/cwa.txt")
    options = CrawlOptions(
        fast_days=7.0,
        hourly_days=7.0,
        daily_days=7.0,
        slow_days=7.0,
        max_failures=12,
        since=None,
        output_root=Path("report_images"),
        dry_run=False,
    )
    run(document_path, options)


if __name__ == "__main__":
    main()

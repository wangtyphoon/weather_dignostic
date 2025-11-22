"""Crawler for NCDR imagery defined in document/ncdr.txt."""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
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
    "Referer": "https://watch.ncdr.nat.gov.tw/",
}

TIMESTAMP_REGEXES: Sequence[re.Pattern[str]] = (
    re.compile(r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}"),
    re.compile(r"\d{4}-\d{4}-\d{4}"),
    re.compile(r"\d{4}-\d{2}-\d{2}_\d{4}"),
    re.compile(r"\d{14}"),
    re.compile(r"\d{12}"),
    re.compile(r"\d{10}"),
    re.compile(r"\d{8}"),
    re.compile(r"(?<!\d)\d{6}(?!\d)"),
)

FORMAT_BY_LENGTH: Dict[int, Sequence[str]] = {
    16: ("%Y-%m-%d-%H-%M",),
    15: ("%Y-%m-%d_%H%M",),
    14: ("%Y%m%d%H%M%S", "%Y-%m%d-%H%M"),
    12: ("%Y%m%d%H%M",),
    10: ("%Y%m%d%H", "%d%m%y%H%M"),
    8: ("%y%m%d%H", "%Y%m%d"),
    6: ("%Y%m", "%y%m%d", "%m%d%H", "%H%M%S"),
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

ANALYSIS_KEYWORDS: Sequence[str] = ("ecmwf", "analysis", "??")


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


def apply_timestamp(patterns: Sequence["TimestampPattern"], template: str, ts: dt.datetime) -> str:
    if not patterns:
        return template
    parts: List[str] = []
    cursor = 0
    for pattern in patterns:
        parts.append(template[cursor:pattern.start])
        if pattern.fmt:
            parts.append(pattern.render(ts))
        else:
            parts.append(pattern.token)
        cursor = pattern.end
    parts.append(template[cursor:])
    return "".join(parts)


def pattern_precision(fmt: str) -> int:
    precision = 0
    if "%S" in fmt:
        precision = max(precision, 5)
    if "%M" in fmt:
        precision = max(precision, 4)
    if "%H" in fmt:
        precision = max(precision, 3)
    if "%d" in fmt:
        precision = max(precision, 2)
    if any(token in fmt for token in ("%m", "%b", "%B")):
        precision = max(precision, 1)
    return precision



def select_render_pattern(patterns: Sequence["TimestampPattern"]) -> Optional["TimestampPattern"]:
    renderable = [pattern for pattern in patterns if pattern.fmt]
    if not renderable:
        return None
    return max(
        renderable,
        key=lambda pattern: (pattern_precision(pattern.fmt or ""), len(pattern.token), -pattern.start),
    )



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


def extract_timestamp_patterns(url: str) -> List[TimestampPattern]:
    patterns: List[TimestampPattern] = []
    seen: Set[Tuple[int, int]] = set()
    covered: List[Tuple[int, int]] = []
    for regex in TIMESTAMP_REGEXES:
        for match in regex.finditer(url):
            start, end = match.start(), match.end()
            key = (start, end)
            if key in seen:
                continue
            if any(s <= start and end <= e for s, e in covered):
                continue
            token = match.group(0)
            fmt = guess_datetime_format(token)
            patterns.append(TimestampPattern(token=token, fmt=fmt, start=start, end=end))
            seen.add(key)
            covered.append(key)
    patterns.sort(key=lambda p: p.start)
    return patterns


@dataclass
class Dataset:
    category: str
    label: str
    url: str
    patterns: List[TimestampPattern] = field(default_factory=list)
    render_pattern: Optional[TimestampPattern] = field(init=False, default=None)

    def __post_init__(self) -> None:
        if not self.patterns:
            self.patterns = extract_timestamp_patterns(self.url)
        self.patterns.sort(key=lambda p: p.start)
        self.render_pattern = select_render_pattern(self.patterns)

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



def uses_utc_clock(dataset: Dataset, interval: dt.timedelta) -> bool:
    url_lower = dataset.url.lower()
    label_lower = dataset.label.lower()
    if "wrf_5km_precipitation" in url_lower or "05days" in url_lower:
        return True
    if interval >= dt.timedelta(hours=12):
        if any(keyword in url_lower for keyword in ANALYSIS_KEYWORDS):
            return True
        if any(keyword in label_lower for keyword in ("analysis",)):
            return True
        if "??" in dataset.label:
            return True
    return False


@dataclass
class CrawlOptions:
    hourly_days: float
    twelve_hour_days: float
    max_failures: int
    since: Optional[dt.datetime]
    output_root: Path
    dry_run: bool


def determine_interval(dataset: Dataset) -> dt.timedelta:
    label = dataset.label
    url_lower = dataset.url.lower()
    if any(keyword in url_lower for keyword in ("wrf_5km_precipitation", "05days")):
        return dt.timedelta(hours=12)
    if "風場" in label or "風" in label and "場" in label:
        return dt.timedelta(hours=1)
    if "windmap" in url_lower or "wind" in url_lower:
        return dt.timedelta(hours=1)
    return dt.timedelta(hours=12)


def compute_lookback(interval: dt.timedelta, options: CrawlOptions) -> dt.timedelta:
    if interval <= dt.timedelta(hours=1):
        return dt.timedelta(days=options.hourly_days)
    return dt.timedelta(days=options.twelve_hour_days)


def reference_now(dataset: Dataset, interval: dt.timedelta) -> dt.datetime:
    if uses_utc_clock(dataset, interval):
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
    datasets: List[Dataset] = []
    category = "NCDR"
    pending_label: Optional[str] = None
    label_consumed = False
    for line in lines:
        if not line or line.startswith("#"):
            continue
        if line[0].isdigit() and "." in line:
            dot = line.find(".")
            pending_label = line[dot + 1 :].strip() or None
            label_consumed = False
            continue
        if line.startswith("http"):
            if label_consumed:
                continue
            url = line
            label = pending_label or Path(url.split("?", 1)[0]).stem
            datasets.append(Dataset(category, label, url))
            if pending_label:
                label_consumed = True
            continue
        category = line
        pending_label = None
        label_consumed = False
    return datasets


def create_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(SESSION_HEADERS)
    return session


def crawl_dataset(session: requests.Session, dataset: Dataset, options: CrawlOptions) -> Tuple[str, str]:
    out_dir = dataset.output_dir(options.output_root)
    patterns = dataset.patterns
    render_pattern = dataset.render_pattern
    if (not patterns or not render_pattern):
        status, message, _ = download_image(session, dataset.url, out_dir, "", options.dry_run)
        return status, message

    interval = determine_interval(dataset)
    lookback = compute_lookback(interval, options)
    seconds = max(int(interval.total_seconds()), 1)
    max_steps = max(int(lookback.total_seconds() // seconds) + 1, 1)
    anchor_base = reference_now(dataset, interval)
    anchor = align_time(anchor_base, interval)

    tokens = existing_tokens(out_dir)
    consecutive_failures = 0
    downloaded = 0
    last_error: Optional[str] = None
    last_token: Optional[str] = None

    for step in range(max_steps):
        candidate_time = anchor - step * interval
        if options.since and candidate_time < options.since:
            break
        token_value = render_pattern.render(candidate_time)
        if token_value in tokens:
            continue
        candidate_url = apply_timestamp(patterns, dataset.url, candidate_time)
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


DOCUMENT_PATH = Path("document/ncdr.txt")
DEFAULT_OPTIONS = CrawlOptions(
    hourly_days=7.0,
    twelve_hour_days=30.0,
    max_failures=12,
    since=None,
    output_root=Path("report_images"),
    dry_run=False,
)


def main() -> None:
    run(DOCUMENT_PATH, DEFAULT_OPTIONS)


if __name__ == "__main__":
    main()

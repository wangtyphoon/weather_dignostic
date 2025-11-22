"""Crawler for CIMSS imagery defined in document/cimss.txt."""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import requests

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None
# 這兩個標籤不需建立時間子資料夾（避免覆蓋的例外清單）
EXEMPT_LABELS: Set[str] = {"可降水量", "颱風高層風場"}

def _hour_bucket_for(dataset: "Dataset") -> str:
    """用資料集的參考時區產生小時粒度時間桶。"""
    now = reference_now(dataset)
    # 以 reference_now 的時區為準（CIMSS 會是 UTC）
    return now.strftime("%Y%m%d%H")

def output_dir_for(dataset: "Dataset", options: "CrawlOptions") -> Path:
    """
    決定實際輸出資料夾：
    - 若 label 在 EXEMPT_LABELS：不加時間子資料夾
    - 其它：加上小時粒度的時間子資料夾（避免相同檔名互相覆蓋）
    """
    base = dataset.output_dir(options.output_root)
    if dataset.label in EXEMPT_LABELS:
        return base
    return base / _hour_bucket_for(dataset)
TZ = ZoneInfo("Asia/Taipei") if ZoneInfo else dt.timezone(dt.timedelta(hours=8))

SESSION_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
    "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
    "Referer": "https://tropic.ssec.wisc.edu/",
}

DEFAULT_TIMEOUT: Tuple[int, int] = (10, 30)

_SANITIZE_PATTERN = re.compile(r'[<>:"/\|?*]')
_GENERAL_TOKEN_REGEX = re.compile(r"\d{4,16}")

FORMAT_BY_LENGTH: Dict[int, Sequence[str]] = {
    16: ("%Y-%m-%d-%H-%M",),
    15: ("%Y-%m-%d_%H%M",),
    14: ("%Y%m%d%H%M%S", "%Y-%m%d-%H%M"),
    12: ("%Y%m%d%H%M",),
    10: ("%Y%m%d%H", "%d%m%y%H%M"),
    8: ("%Y%m%d", "%y%m%d%H"),
}

FALLBACK_FORMATS: Sequence[str] = (
    "%Y%m%d%H%M%S",
    "%Y%m%d%H%M",
    "%Y%m%d%H",
    "%Y%m%d",
    "%Y%m",
    "%Y",
    "%H%M%S",
    "%H%M",
)

TOKEN_DELIM = "__"


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


@dataclass
class Dataset:
    category: str
    label: str
    url: str
    patterns: List[TimestampPattern] = field(default_factory=list)
    renderable_patterns: List[TimestampPattern] = field(init=False, default_factory=list)
    static_sequence: Optional[List[str]] = field(init=False, default=None)
    sample_datetime: Optional[dt.datetime] = field(init=False, default=None)
    token_format: Optional[str] = field(init=False, default=None)

    def __post_init__(self) -> None:
        if not self.patterns:
            self.patterns = extract_timestamp_patterns(self.url)
        self.patterns.sort(key=lambda p: p.start)
        self.renderable_patterns = [pattern for pattern in self.patterns if pattern.fmt]
        self.static_sequence = self._compute_static_sequence()
        self.token_format = self._compute_token_format()
        self.sample_datetime = self._compute_sample_datetime()
        self._primary_pattern = self._select_primary_pattern()

    def output_dir(self, root: Path) -> Path:
        return root / sanitize(self.category) / sanitize(self.label)

    def primary_pattern(self) -> Optional[TimestampPattern]:
        return self._primary_pattern

    def token_for(self, ts: dt.datetime) -> str:
        if self.token_format:
            return ts.strftime(self.token_format)
        if not self.renderable_patterns:
            return ""
        return "_".join(pattern.render(ts) for pattern in self.renderable_patterns)

    def datetime_from_token(self, token: str) -> Optional[dt.datetime]:
        if not token:
            return None
        if self.token_format:
            try:
                parsed = dt.datetime.strptime(token, self.token_format)
            except ValueError:
                pass
            else:
                return parsed.replace(tzinfo=TZ)
        if not self.renderable_patterns:
            return None
        segments = token.split("_")
        if len(segments) != len(self.renderable_patterns):
            return None
        return self._compose_datetime(segments)

    def latest_token_time(self, tokens: Iterable[str]) -> Optional[dt.datetime]:
        latest: Optional[dt.datetime] = None
        for token in tokens:
            candidate = self.datetime_from_token(token)
            if candidate and (latest is None or candidate > latest):
                latest = candidate
        return latest

    def render_url(self, ts: dt.datetime) -> str:
        if not self.patterns:
            return self.url
        parts: List[str] = []
        last = 0
        for pattern in self.patterns:
            parts.append(self.url[last:pattern.start])
            parts.append(pattern.render(ts))
            last = pattern.end
        parts.append(self.url[last:])
        return "".join(parts)

    def _select_primary_pattern(self) -> Optional[TimestampPattern]:
        if not self.patterns:
            return None
        return max(self.patterns, key=_pattern_priority)

    def _compute_static_sequence(self) -> Optional[List[str]]:
        match = re.search(r"^(?P<base>.*?)(?:-(?P<index>\d+))(?P<ext>\.[^.]+)$", self.url)
        if not match:
            return None
        index = match.group("index")
        if index != "1":
            return None
        base = match.group("base")
        ext = match.group("ext")
        sequence = [f"{base}{ext}"]
        sequence.extend(f"{base}-{offset}{ext}" for offset in range(1, 9))
        return sequence

    def _compute_token_format(self) -> Optional[str]:
        if not self.renderable_patterns:
            return None
        has_year = has_month = has_day = False
        has_hour = has_minute = has_second = False
        for pattern in self.renderable_patterns:
            fmt = pattern.fmt or ""
            if "%Y" in fmt or "%y" in fmt:
                has_year = True
            if "%m" in fmt:
                has_month = True
            if "%d" in fmt:
                has_day = True
            if "%H" in fmt:
                has_hour = True
            if "%M" in fmt:
                has_minute = True
            if "%S" in fmt:
                has_second = True
        token_fmt = ""
        if has_year:
            token_fmt += "%Y"
        if has_month:
            token_fmt += "%m"
        if has_day:
            token_fmt += "%d"
        if has_hour:
            token_fmt += "%H"
        if has_minute:
            token_fmt += "%M"
        if has_second:
            token_fmt += "%S"
        if token_fmt:
            return token_fmt
        # fallback to the longest available format if nothing else matches
        longest = max(self.renderable_patterns, key=lambda p: p.end - p.start)
        return longest.fmt

    def _compose_datetime(self, values: Iterable[str]) -> Optional[dt.datetime]:
        components = {"year": None, "month": None, "day": None, "hour": None, "minute": None, "second": None}
        for value, pattern in zip(values, self.renderable_patterns):
            fmt = pattern.fmt
            if not fmt:
                continue
            try:
                parsed = dt.datetime.strptime(value, fmt)
            except ValueError:
                return None
            if ("%Y" in fmt or "%y" in fmt) and components["year"] is None:
                components["year"] = parsed.year
            if "%m" in fmt and components["month"] is None:
                components["month"] = parsed.month
            if "%d" in fmt and components["day"] is None:
                components["day"] = parsed.day
            if "%H" in fmt and components["hour"] is None:
                components["hour"] = parsed.hour
            if "%M" in fmt and components["minute"] is None:
                components["minute"] = parsed.minute
            if "%S" in fmt and components["second"] is None:
                components["second"] = parsed.second
        year = components["year"]
        month = components["month"] or 1
        day = components["day"] or 1
        hour = components["hour"] or 0
        minute = components["minute"] or 0
        second = components["second"] or 0
        if year is None:
            return None
        return dt.datetime(year, month, day, hour, minute, second, tzinfo=TZ)

    def _compute_sample_datetime(self) -> Optional[dt.datetime]:
        if not self.renderable_patterns:
            return None
        values = [pattern.token for pattern in self.renderable_patterns]
        return self._compose_datetime(values)


def sanitize(text: str) -> str:
    cleaned = _SANITIZE_PATTERN.sub("_", text.strip())
    return cleaned or "untitled"


def parse_document(path: Path) -> List[Dataset]:
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    datasets: List[Dataset] = []
    category: Optional[str] = None
    pending_label: Optional[str] = None
    last_dataset: Optional[Dataset] = None
    for line in lines:
        if not line:
            continue
        if line.startswith("http"):
            if not category:
                continue
            if pending_label is None and last_dataset and last_dataset.category == category:
                continue
            label = pending_label or Path(line.split("?", 1)[0]).stem
            dataset = Dataset(category=category, label=label, url=line)
            datasets.append(dataset)
            last_dataset = dataset
            pending_label = None
            continue
        if line[0].isdigit() and "." in line:
            dot = line.find(".")
            pending_label = line[dot + 1 :].strip() or None
            last_dataset = None
            continue
        category = line
        pending_label = None
        last_dataset = None
    return datasets


def create_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(SESSION_HEADERS)
    return session


def existing_tokens(out_dir: Path) -> Set[str]:
    tokens: Set[str] = set()
    if not out_dir.exists():
        return tokens
    for path in out_dir.iterdir():
        if not path.is_file() or path.name.startswith("_"):
            continue
        name = path.name
        if TOKEN_DELIM in name:
            tokens.add(name.split(TOKEN_DELIM, 1)[0])
        elif "_" in name:
            tokens.add(name.split("_", 1)[0])
    return tokens


def align_time(ts: dt.datetime, interval: dt.timedelta) -> dt.datetime:
    seconds = max(int(interval.total_seconds()), 1)
    aligned_epoch = int(ts.timestamp()) - (int(ts.timestamp()) % seconds)
    return dt.datetime.fromtimestamp(aligned_epoch, tz=ts.tzinfo)


def determine_interval(dataset: Dataset, pattern: TimestampPattern) -> dt.timedelta:
    label = dataset.label
    url_lower = dataset.url.lower()

    # --- NEW: CIMSS mesoamv / HIMOFcombined are not aligned to 5/10 minutes reliably
    if "mesoamv" in url_lower or "himofcombined" in url_lower:
        return dt.timedelta(minutes=10)

    if "可降水量" in label:
        return dt.timedelta(hours=1)
    if "颱風高層風場" in label:
        return dt.timedelta(hours=1)

    if "radar_rain" in url_lower:
        return dt.timedelta(minutes=5)
    if "radar" in url_lower:
        return dt.timedelta(minutes=10)
    if "satellite" in url_lower:
        return dt.timedelta(minutes=10)
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
    if "%m" in fmt:
        return dt.timedelta(days=31)
    return dt.timedelta(days=7)


def reference_now(dataset: Dataset) -> dt.datetime:
    url_lower = dataset.url.lower()
    # --- NEW: CIMSS site uses UTC timestamps in filenames
    if "tropic.ssec.wisc.edu" in url_lower:
        return dt.datetime.now(dt.timezone.utc)
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
    filename = Path(url.split("?", 1)[0]).name or "download"
    needs_prefix = bool(token)
    dest_name = f"{token}{TOKEN_DELIM}{filename}" if needs_prefix else filename
    dest = out_dir / dest_name
    if dest.exists() and not dry_run:
        return "skip", f"exists {dest.name}", dest
    if dry_run:
        return "ok", f"would download {dest.name}", dest
    try:
        with session.get(url, stream=True, timeout=DEFAULT_TIMEOUT) as resp:
            if resp.status_code != 200:
                return "fail", f"HTTP {resp.status_code}", dest
            content_type = resp.headers.get("Content-Type", "")
            if content_type and "image" not in content_type.lower():
                return "fail", f"unexpected content: {content_type}", dest
            with dest.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        fh.write(chunk)
    except requests.RequestException as exc:
        return "fail", f"request error: {exc}", dest
    return "ok", f"downloaded {dest.name}", dest


def crawl_static_sequence(
    session: requests.Session,
    dataset: Dataset,
    options: "CrawlOptions",
) -> Tuple[str, str]:
    # 原本：out_dir = dataset.output_dir(options.output_root)
    out_dir = output_dir_for(dataset, options)  # ← 改為加上小時子資料夾（除非在例外清單）
    outcomes = {"ok": 0, "skip": 0, "fail": 0}
    last_error: Optional[str] = None
    for seq_url in dataset.static_sequence or []:
        token = Path(seq_url).stem
        status, message, _ = download_image(session, seq_url, out_dir, token, options.dry_run)
        outcomes[status] = outcomes.get(status, 0) + 1
        if status == "fail":
            last_error = message
        print(f"    [{Path(seq_url).name}] {message}")
    if outcomes["fail"]:
        return "fail", last_error or "static sequence failures"
    if outcomes["ok"]:
        note = "dry-run" if options.dry_run else "downloaded"
        return "ok", f"{note} {outcomes['ok']} frames"
    return "skip", "no new files"

def align_time_with_offset(ts: dt.datetime, interval: dt.timedelta, offset_seconds: int) -> dt.datetime:
    """
    將 ts 對齊到間隔 interval 的格點，但加上固定秒數偏移（如每10分鐘的第4分 → offset=240）。
    """
    period = max(int(interval.total_seconds()), 1)
    s = int(ts.timestamp())
    aligned = ((s - offset_seconds) // period) * period + offset_seconds
    return dt.datetime.fromtimestamp(aligned, tz=ts.tzinfo)
def crawl_dataset(session: requests.Session, dataset: Dataset, options: "CrawlOptions") -> Tuple[str, str]:
    if dataset.static_sequence:
        return crawl_static_sequence(session, dataset, options)

    # 原本：out_dir = dataset.output_dir(options.output_root)
    out_dir = output_dir_for(dataset, options)  # ← 改為加上小時子資料夾（除非在例外清單）
    tokens = existing_tokens(out_dir)
    primary = dataset.primary_pattern()
    if not dataset.renderable_patterns or not primary or not primary.fmt:
        status, message, _ = download_image(session, dataset.url, out_dir, "", options.dry_run)
        return status, message

    interval = determine_interval(dataset, primary)
    dataset_now = reference_now(dataset)
    tzinfo = dataset_now.tzinfo or TZ
    # enforce a one-week window for downloads by default
    since_limit = dataset_now - dt.timedelta(days=7)
    raw_since = options.since
    if raw_since:
        if raw_since.tzinfo is None:
            raw_since = raw_since.replace(tzinfo=tzinfo)
        if raw_since > since_limit:
            since_limit = raw_since

    anchor_base = dataset_now
    latest_existing = dataset.latest_token_time(tokens)
    if latest_existing and latest_existing > anchor_base:
        anchor_base = latest_existing
    elif not latest_existing and dataset.sample_datetime and dataset.sample_datetime > since_limit:
        anchor_base = dataset.sample_datetime
    if anchor_base < since_limit:
        anchor_base = since_limit

    url_lower = dataset.url.lower()
    if "mesoamv" in url_lower or "himofcombined" in url_lower:
        # 每10分鐘、但固定 +4 分鐘對齊
        anchor = align_time_with_offset(anchor_base, interval, offset_seconds=4 * 60)
    else:
        anchor = align_time(anchor_base, interval)

    while anchor < since_limit:
        next_anchor = anchor + interval
        if next_anchor > anchor_base:
            anchor = anchor_base
            break
        anchor = next_anchor

    consecutive_failures = 0
    downloaded = 0
    last_error: Optional[str] = None
    last_token: Optional[str] = None
    step = 0

    while consecutive_failures < options.max_failures:
        try:
            candidate_time = anchor - step * interval
        except OverflowError:
            break
        if candidate_time < since_limit:
            break
        if candidate_time.year < 1900:
            break
        token_value = dataset.token_for(candidate_time)
        if token_value and token_value in tokens:
            step += 1
            continue
        candidate_url = dataset.render_url(candidate_time)
        status, message, _ = download_image(
            session,
            candidate_url,
            out_dir,
            token_value,
            options.dry_run,
        )
        if status == "ok":
            downloaded += 1
            if token_value:
                tokens.add(token_value)
            consecutive_failures = 0
            last_token = token_value or last_token
        elif status == "skip":
            if token_value:
                tokens.add(token_value)
            consecutive_failures = 0
        else:
            last_error = message
            consecutive_failures += 1
        if status != "ok":
            last_error = message
        step += 1

    if downloaded:
        tail = f" (latest token {last_token})" if last_token else ""
        note = "dry-run" if options.dry_run else "downloaded"
        return "ok", f"{note} {downloaded} files{tail}"
    if tokens:
        return "skip", "no new files"
    return "fail", last_error or "no images reachable"

def run(document_path: Path, options: "CrawlOptions") -> None:
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


@dataclass
class CrawlOptions:
    output_root: Path
    dry_run: bool
    max_failures: int
    since: Optional[dt.datetime] = None


def _pattern_priority(pattern: TimestampPattern) -> Tuple[int, int, int, int, int, int, int]:
    fmt = pattern.fmt or ""
    return (
        1 if ("%Y" in fmt or "%y" in fmt) else 0,
        1 if "%m" in fmt else 0,
        1 if "%d" in fmt else 0,
        1 if "%H" in fmt else 0,
        1 if "%M" in fmt else 0,
        1 if "%S" in fmt else 0,
        -(pattern.end - pattern.start),
    )


def guess_datetime_format(token: str, *, context: str = "", is_path_segment: bool = False) -> Optional[str]:
    """
    Heuristics with context:
    - If token length==4 and <=2359 and context contains an 8-digit date in the same filename
      (e.g., '...20250922.1714...'), prefer %H%M over %Y.
    - If token length==4 and appears as a standalone path segment like '/2025/', prefer %Y.
    - Otherwise fall back to numeric rules.
    """
    length = len(token)

    def has_8digit_date(s: str) -> bool:
        return bool(re.search(r"(?<!\d)(\d{8})(?!\d)", s))

    # Special handling for 4-digit tokens
    if length == 4 and token.isdigit():
        val = int(token)
        # If same component contains an 8-digit date, the 4-digit is very likely HHMM.
        if has_8digit_date(context) and 0 <= val <= 2359:
            return "%H%M"
        # If it's a standalone path segment like '/2025/', treat as year.
        if is_path_segment and 1900 <= val <= 2100:
            return "%Y"
        # Otherwise try %H%M first, then %Y.
        for fmt in ("%H%M", "%Y", "%m%d"):
            try:
                dt.datetime.strptime(token, fmt)
                return fmt
            except ValueError:
                pass
        # Fallback to old logic (rare)
        return None

    # 6-digit ambiguities: try %H%M%S after %Y%m and %y%m%d
    if length == 6 and token.isdigit():
        for fmt in ("%Y%m", "%y%m%d", "%H%M%S"):
            try:
                dt.datetime.strptime(token, fmt)
                return fmt
            except ValueError:
                pass
        # Then try configured tables and fallbacks
        for fmt in FORMAT_BY_LENGTH.get(length, ()):
            try:
                dt.datetime.strptime(token, fmt)
                return fmt
            except ValueError:
                pass
        for fmt in FALLBACK_FORMATS:
            try:
                dt.datetime.strptime(token, fmt)
                return fmt
            except ValueError:
                pass
        return None

    # Default: use existing tables then fallbacks
    for fmt in FORMAT_BY_LENGTH.get(length, ()):
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
    """
    Extract tokens with local filename/path context to disambiguate 4-digit tokens.
    """
    patterns: List[TimestampPattern] = []

    # Identify the basename component and the immediate path segment of each match
    # so we can pass context signals (e.g., 'HIMOFcombined.20250922.1714.GIF').
    # We'll also mark if the token itself is a full path segment like '/2025/'.
    # Pre-split path:
    parts = url.split("/")
    # Build ranges for quick "which segment?" lookup
    seg_ranges: List[Tuple[int, int, str]] = []
    idx = 0
    for seg in parts:
        start = idx
        end = idx + len(seg)
        seg_ranges.append((start, end, seg))
        idx = end + 1  # account for '/'

    for match in _GENERAL_TOKEN_REGEX.finditer(url):
        token = match.group(0)
        start, end = match.start(), match.end()

        # find containing segment
        seg_text = ""
        is_seg = False
        for s, e, seg in seg_ranges:
            if s <= start <= e:
                seg_text = seg
                # standalone path segment if entire segment equals token
                is_seg = (seg == token)
                break

        fmt = guess_datetime_format(token, context=seg_text, is_path_segment=is_seg)
        if not fmt:
            continue
        patterns.append(TimestampPattern(token=token, fmt=fmt, start=start, end=end))
    return patterns

DOCUMENT_PATH = Path("document/cimss.txt")
OUTPUT_ROOT = Path("report_images")
OPTIONS = CrawlOptions(output_root=OUTPUT_ROOT, dry_run=False, max_failures=48)


def main() -> None:
    run(DOCUMENT_PATH, OPTIONS)


if __name__ == "__main__":
    main()

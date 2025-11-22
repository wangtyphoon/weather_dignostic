#!/usr/bin/env python3
"""Download Tropical Tidbits ECMWF images listed in document/TT.txt.

The script replaces the timestamp portion of each URL (yyyyMMddHH) with the
most recent 6-hourly cycle based on the current UTC time and walks backward in
6-hour steps automatically. It stops when it reaches a cycle that is already
fully downloaded or after hitting a configurable number of 404 cycles.
"""

from __future__ import annotations

import datetime as dt
import os
import re
import sys
from typing import Iterable, List
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

TIMESTAMP_RE = re.compile(r"/\d{10}/")

# Configuration
URL_FILE = os.path.join("document", "TT.txt")
OUT_DIR = "tt_images"
INTERVAL_HOURS = 6
LAG_HOURS = 0  # shift latest cycle backward if the newest data is not ready
MAX_CYCLES = 40  # safety cap to avoid infinite loops
ALLOWED_404_CYCLES = 2  # stop after this many 404-hit cycles
FORCE = False  # set True to redownload existing files


def read_urls(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"URL list not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def timestamped_url(url: str, timestamp: str) -> str | None:
    """Replace the first yyyyMMddHH segment in the URL."""
    if not TIMESTAMP_RE.search(url):
        return None
    return TIMESTAMP_RE.sub(f"/{timestamp}/", url, count=1)


def fetch(url: str, dest: str, force: bool = False) -> str:
    if os.path.exists(dest) and not force:
        print(f"= exists: {dest}")
        return "exists"

    os.makedirs(os.path.dirname(dest), exist_ok=True)

    req = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://www.tropicaltidbits.com/analysis/models/",
        },
    )
    try:
        with urlopen(req) as resp:
            data = resp.read()
    except HTTPError as exc:
        if exc.code == 404:
            print(f"! 404 not found: {url}")
            return "not_found"
        print(f"! failed ({exc}): {url}")
        return "error"
    except URLError as exc:
        print(f"! failed ({exc}): {url}")
        return "error"

    with open(dest, "wb") as f:
        f.write(data)
    print(f"+ saved: {dest}")
    return "saved"


def cycle_times(now: dt.datetime, interval_hours: int) -> Iterable[dt.datetime]:
    """Yield cycle datetimes starting from the latest toward the past."""
    rounded_hours = (now.hour // interval_hours) * interval_hours
    latest = now.replace(hour=rounded_hours, minute=0, second=0, microsecond=0)
    step = 0
    while step < MAX_CYCLES:
        yield latest - dt.timedelta(hours=interval_hours * step)
        step += 1


def main() -> int:
    urls = read_urls(URL_FILE)

    now = dt.datetime.utcnow() - dt.timedelta(hours=LAG_HOURS)
    print(f"Using current UTC time: {dt.datetime.utcnow():%Y-%m-%d %H:%M}")
    if LAG_HOURS:
        print(f"Applying lag of {LAG_HOURS} hour(s): {now:%Y-%m-%d %H:%M}")
    print(
        f"Interval: {INTERVAL_HOURS}h; output: {OUT_DIR}; max cycles: {MAX_CYCLES}; "
        f"force: {FORCE}; 404 cycles allowed: {ALLOWED_404_CYCLES}"
    )

    not_found_cycles = 0
    for cycle_time in cycle_times(now, INTERVAL_HOURS):
        ts = cycle_time.strftime("%Y%m%d%H")
        print(f"--- Cycle {ts} ---")
        all_exist = True
        hit_not_found = False

        for url in urls:
            final_url = timestamped_url(url, ts)
            if not final_url:
                print(f"! no timestamp placeholder found, skipped: {url}")
                continue
            filename = os.path.basename(final_url)
            dest = os.path.join(OUT_DIR, ts, filename)
            result = fetch(final_url, dest, force=FORCE)
            if result != "exists":
                all_exist = False
            if result == "not_found":
                hit_not_found = True

        if hit_not_found:
            not_found_cycles += 1
            print(f"Cycle {ts} had 404s ({not_found_cycles}/{ALLOWED_404_CYCLES})")
            if not_found_cycles >= ALLOWED_404_CYCLES:
                print(f"Stopping: reached {not_found_cycles} cycle(s) with 404s")
                break
        else:
            not_found_cycles = 0
        if all_exist:
            print(f"Stopping: cycle {ts} already downloaded")
            break

    return 0


if __name__ == "__main__":
    sys.exit(main())

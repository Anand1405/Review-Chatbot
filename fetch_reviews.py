from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path
from time import sleep
from typing import List, Optional, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry

try:
    from google_play_scraper import Sort, reviews as gp_reviews
except ImportError as exc:  
    raise ImportError(
        "google_play_scraper is not installed. Run 'pip install google_play_scraper'."
    ) from exc

try:
    from app_store_scraper import AppStore
except ImportError:
    AppStore = None


def fetch_google_play_reviews(
    app_id: str,
    count: int,
    lang: str = "en",
    country: str = "in",
    delay: float = 0.0,
) -> pd.DataFrame:
    """Fetch up to ``count`` Google Play reviews."""

    collected: List[dict] = []
    continuation_token: Optional[Tuple[str, str]] = None

    while len(collected) < count:
        batch_size = min(200, count - len(collected))
        batch, continuation_token = gp_reviews(
            app_id,
            lang=lang,
            country=country,
            sort=Sort.NEWEST,
            count=batch_size,
            continuation_token=continuation_token,
        )
        if not batch:
            break
        collected.extend(batch)
        print(f"[Google Play] Fetched {len(collected)} reviews...", file=sys.stderr)
        if continuation_token is None:
            break
        if delay:
            sleep(delay)

    if not collected:
        return pd.DataFrame(columns=["id", "rating", "text", "date", "version", "platform"])

    df = pd.DataFrame(collected).rename(
        columns={
            "reviewId": "id",
            "score": "rating",
            "content": "text",
            "at": "date",
            "reviewCreatedVersion": "version",
        }
    )
    df["version"] = df["version"].fillna("unknown").astype(str)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["platform"] = "Android"
    return df[["id", "rating", "text", "date", "version", "platform"]]


def _session_with_retries() -> requests.Session:
    session = requests.Session()
    retry_kwargs = {
        "total": 5,
        "backoff_factor": 0.6,
        "status_forcelist": (429, 500, 502, 503, 504),
        "raise_on_redirect": False,
        "raise_on_status": False,
    }
    try:
        retries = Retry(allowed_methods=("GET",), **retry_kwargs)  [arg-type]
    except TypeError:  # compatibility with urllib3<1.26
        retries = Retry(method_whitelist=("GET",), **retry_kwargs)  [arg-type]
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.headers.update(
        {
            "User-Agent": "LiquideReviewAgent/1.0 (+https://liquide.trading)",
            "Accept": "application/json",
        }
    )
    return session


def _fetch_app_store_api(
    app_id: int,
    app_name: str,
    count: int,
    country: str,
) -> List[dict]:
    if AppStore is None or not app_name:
        return []
    scraper = AppStore(country=country, app_name=app_name, app_id=app_id)
    try:
        scraper.review(how_many=count)
    except Exception as exc:  # pragma: no cover - external dependency
        print(f"[App Store API] Failed: {exc}", file=sys.stderr)
        return []
    rows: List[dict] = []
    for item in scraper.reviews or []:
        review_id = item.get("id") or item.get("reviewId") or str(uuid.uuid4())
        rating = int(item.get("rating", 0))
        text = item.get("review", "").strip()
        date = item.get("date")
        if hasattr(date, "strftime"):
            date_str = date.strftime("%Y-%m-%d")
        else:
            date_str = pd.to_datetime(date).strftime("%Y-%m-%d") if date else ""
        version = str(item.get("version", "unknown"))
        rows.append(
            {
                "id": review_id,
                "rating": rating,
                "text": text,
                "date": date_str,
                "version": version,
                "platform": "iOS",
            }
        )
    return rows


def _fetch_app_store_rss(
    app_id: int,
    count: int,
    country: str,
    language: str,
    delay: float,
) -> List[dict]:
    session = _session_with_retries()
    collected: List[dict] = []
    seen_ids = set()
    page = 1
    country_path = country.lower()

    while len(collected) < count:
        url = (
            f"https://itunes.apple.com/{country_path}/rss/customerreviews/id={app_id}"
            f"/page={page}/sortby=mostrecent/json"
        )
        params = {"l": language}
        response = session.get(url, params=params, timeout=15)
        if response.status_code != 200:
            print(
                f"[App Store RSS] Request failed (status {response.status_code}) on page {page}",
                file=sys.stderr,
            )
            break
        payload = response.json()
        entries = payload.get("feed", {}).get("entry", [])
        if len(entries) <= 1:
            if page == 1:
                print("[App Store RSS] No review entries returned.", file=sys.stderr)
            break

        new_count = 0
        for entry in entries[1:]:
            review_id = entry.get("id", {}).get("label")
            if not review_id or review_id in seen_ids:
                continue
            seen_ids.add(review_id)
            rating_raw = entry.get("im:rating", {}).get("label", "0")
            try:
                rating = int(rating_raw)
            except (TypeError, ValueError):
                rating = 0
            text = entry.get("content", {}).get("label", "").strip()
            updated = entry.get("updated", {}).get("label") or entry.get("im:releaseDate", {}).get("label")
            version = entry.get("im:version", {}).get("label", "unknown")
            collected.append(
                {
                    "id": review_id,
                    "rating": rating,
                    "text": text,
                    "date": pd.to_datetime(updated).strftime("%Y-%m-%d") if updated else "",
                    "version": str(version),
                    "platform": "iOS",
                }
            )
            new_count += 1
            if len(collected) >= count:
                break

        print(
            f"[App Store RSS] Fetched {new_count} new reviews (total {len(collected)}) on page {page}...",
            file=sys.stderr,
        )
        if new_count == 0:
            break
        page += 1
        if delay:
            sleep(delay)

    return collected


def fetch_app_store_reviews(
    app_id: int,
    app_name: str,
    count: int,
    country: str = "us",
    language: str = "en-us",
    delay: float = 0.5,
) -> pd.DataFrame:
    """Fetch reviews from Apple using the API when possible, otherwise RSS."""

    api_rows = _fetch_app_store_api(app_id, app_name, count, country)
    if api_rows:
        source = "API"
        rows = api_rows
    else:
        print("[App Store] Falling back to RSS feed scraping...", file=sys.stderr)
        rows = _fetch_app_store_rss(app_id, count, country, language, delay)
        source = "RSS"

    if not rows:
        print(
            "[App Store] No reviews were collected. The app may not have public reviews in this storefront.",
            file=sys.stderr,
        )
        return pd.DataFrame(columns=["id", "rating", "text", "date", "version", "platform"])

    print(f"[App Store] Retrieved {len(rows)} reviews via {source}.", file=sys.stderr)
    df = pd.DataFrame(rows)
    return df[["id", "rating", "text", "date", "version", "platform"]]


def combine_and_save(
    frames: List[pd.DataFrame],
    output_path: str,
    db_path: Optional[str],
) -> None:
    combined = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    combined = combined.drop_duplicates(subset=["id", "platform"]).reset_index(drop=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(f"Saved {len(combined)} reviews to {output_path}")

    if db_path:
        from ingest_reviews import ingest_data

        ingest_data(output_path, db_path)
        print(f"Ingested {len(combined)} reviews into {db_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Liquide reviews from Google Play and the Apple App Store.")
    parser.add_argument("--android-app-id", help="Android package name (e.g. life.liquide.app).")
    parser.add_argument("--ios-app-id", type=int, help="Numeric Apple App Store identifier (e.g. 1624726081).")
    parser.add_argument(
        "--ios-app-name",
        help="Human-readable Apple App Store name, e.g. 'Liquide- Stocks & Mutual Funds'.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the CSV file where combined reviews will be saved.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=2000,
        help="Maximum reviews to fetch per storefront (default: 2000).",
    )
    parser.add_argument("--lang", default="en", help="Language code for Google Play reviews (default: en).")
    parser.add_argument("--country", default="in", help="Country code for Google Play reviews (default: in).")
    parser.add_argument(
        "--ios-country",
        default="us",
        help="Country code for App Store reviews (default: us).",
    )
    parser.add_argument(
        "--ios-lang",
        default="en-us",
        help="Language locale for App Store reviews (default: en-us).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay (seconds) between paginated requests to avoid rate limits.",
    )
    parser.add_argument("--db", help="Optional SQLite database; if provided the data is ingested automatically.")
    args = parser.parse_args()

    if not args.android_app_id and not args.ios_app_id:
        parser.error("Provide at least one of --android-app-id or --ios-app-id.")

    frames: List[pd.DataFrame] = []

    if args.android_app_id:
        frames.append(
            fetch_google_play_reviews(
                app_id=args.android_app_id,
                count=args.count,
                lang=args.lang,
                country=args.country,
                delay=args.delay,
            )
        )

    if args.ios_app_id:
        frames.append(
            fetch_app_store_reviews(
                app_id=args.ios_app_id,
                app_name=args.ios_app_name or "",
                count=args.count,
                country=args.ios_country,
                language=args.ios_lang,
                delay=args.delay,
            )
        )

    if not frames:
        print("No reviews fetched. Nothing to write.", file=sys.stderr)
        return

    combine_and_save(frames, args.output, args.db)


if __name__ == "__main__":
    main()

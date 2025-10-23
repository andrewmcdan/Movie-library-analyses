from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from queue import Empty, Queue
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Tuple

import PySimpleGUI as sg  # type: ignore

try:
    from plexapi.exceptions import NotFound, Unauthorized
    from plexapi.server import PlexServer
except ImportError as exc:
    sys.stderr.write(
        "Missing dependency: plexapi.\n"
        "Install it with 'pip install plexapi'.\n"
    )
    raise


TRACE_LEVEL_NUM = 5
LOG_LEVELS = {
    "trace": TRACE_LEVEL_NUM,
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warn": logging.WARNING,
    "error": logging.ERROR,
}


def _configure_trace_logging() -> None:
    if hasattr(logging, "TRACE"):
        return

    logging.addLevelName(TRACE_LEVEL_NUM, "TRACE")

    def trace(self, message, *args, **kwargs):
        if self.isEnabledFor(TRACE_LEVEL_NUM):
            self._log(TRACE_LEVEL_NUM, message, args, **kwargs)

    setattr(logging.Logger, "trace", trace)  # type: ignore[attr-defined]


_configure_trace_logging()
logger = logging.getLogger("plex_analytics_gui")


def parse_args(argv: List[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plex movie analytics GUI.")
    parser.add_argument(
        "--log-level",
        choices=LOG_LEVELS.keys(),
        default="info",
        help="Logging level (default: info).",
    )
    parser.add_argument(
        "--theme",
        default="SystemDefault",
        help="PySimpleGUI theme name (default: SystemDefault).",
    )
    return parser.parse_args(argv)


def configure_logging(level_name: str) -> None:
    level = LOG_LEVELS.get(level_name.lower())
    if level is None:
        raise ValueError(f"Unknown log level '{level_name}'.")

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.setLevel(level)
    logger.debug("Logging configured at level %s", level_name.upper())


ENV_FILE = os.path.join(os.path.dirname(__file__), ".env")
REQUIRED_ENV_KEYS = ("PLEX_URL", "PLEX_TOKEN", "PLEX_LIBRARY_NAME")


def load_env_file(path: str) -> None:
    if not os.path.exists(path):
        logger.warning("Environment file %s not found; relying on existing environment.", path)
        return

    with open(path, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            if key and key not in os.environ:
                os.environ[key] = value
                logger.trace("Loaded %s from .env file.", key)


def get_required_env(keys: Iterable[str]) -> Dict[str, str]:
    missing = [key for key in keys if not os.environ.get(key)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")
    env = {key: os.environ[key] for key in keys}
    logger.debug("Using environment variables: %s", ", ".join(env.keys()))
    return env


def connect_to_plex(url: str, token: str) -> PlexServer:
    try:
        logger.info("Connecting to Plex at %s", url)
        server = PlexServer(url, token)
        logger.debug("Connected to Plex server: %s", server.friendlyName)
        return server
    except Unauthorized as err:
        logger.error("Plex authorization failed: %s", err)
        raise RuntimeError("Plex authorization failed. Check PLEX_TOKEN.") from err
    except NotFound as err:
        logger.error("Plex server not reachable: %s", err)
        raise RuntimeError("Plex server could not be reached. Check PLEX_URL.") from err


def fetch_movies(library) -> Iterable:
    try:
        logger.info("Fetching movies from library '%s'", library.title)
        return library.all(libtype="movie")
    except NotFound as err:
        logger.error("Library '%s' unavailable: %s", library.title, err)
        raise RuntimeError(
            f"Plex library '{library.title}' is unavailable or contains no movies."
        ) from err


def extract_provider_from_guid(guid_value: str) -> str:
    if not guid_value:
        return ""

    provider = guid_value.split("://", 1)[0].lower()
    if provider.startswith("com.plexapp.agents."):
        provider = provider.split("com.plexapp.agents.", 1)[1]
    if provider.startswith("com.plexapp.plugins."):
        provider = provider.split("com.plexapp.plugins.", 1)[1]

    provider = provider.split(".")[-1]
    return provider.upper()


def get_primary_audio_language(movie) -> str:
    try:
        stream = movie.defaultAudioStream()
    except Exception:
        stream = None

    def pick_language(stream_obj) -> str:
        if not stream_obj:
            return ""
        return (
            getattr(stream_obj, "language", None)
            or getattr(stream_obj, "languageCode", None)
            or ""
        )

    language = pick_language(stream)
    if language:
        return str(language)

    try:
        streams = movie.audioStreams()
    except Exception:
        streams = []

    for audio_stream in streams or []:
        language = pick_language(audio_stream)
        if language:
            return str(language)

    return ""


def human_readable_duration(minutes_value: float) -> str:
    if not minutes_value:
        return "n/a"
    total_minutes = int(round(minutes_value))
    hours, minutes = divmod(total_minutes, 60)
    return f"{hours}h {minutes}m" if hours else f"{minutes}m"


def pretty_provider(name: str) -> str:
    if not name:
        return "Unknown"
    lookup = {
        "IMDB": "IMDb",
        "TMDB": "TMDb",
        "THEMOVIEDB": "TMDb",
        "LOCAL": "Local Files",
        "PLEX": "Plex",
    }
    upper_name = name.upper()
    return lookup.get(upper_name, upper_name.title())


def format_counter(counter: Counter[str], limit: int = 5, transform=lambda value: value) -> str:
    if not counter:
        return "  n/a"
    lines = []
    for name, count in counter.most_common(limit):
        lines.append(f"  {transform(name)}: {count}")
    return "\n".join(lines)


def format_metadata_gaps(gaps: Dict[str, int]) -> str:
    if not gaps:
        return "  None"
    lines = []
    for key, value in sorted(gaps.items(), key=lambda item: item[1], reverse=True):
        friendly = key.replace("_", " ").title()
        lines.append(f"  {friendly}: {value}")
    return "\n".join(lines)


class AnalysisAccumulator:
    def __init__(self) -> None:
        self.actor_counts: Counter[str] = Counter()
        self.director_counts: Counter[str] = Counter()
        self.genre_counts: Counter[str] = Counter()
        self.content_rating_counts: Counter[str] = Counter()
        self.language_counts: Counter[str] = Counter()
        self.country_counts: Counter[str] = Counter()
        self.collection_counts: Counter[str] = Counter()
        self.resolution_counts: Counter[str] = Counter()
        self.hdr_counts: Counter[str] = Counter()
        self.audio_channel_counts: Counter[str] = Counter()
        self.source_counts: Counter[str] = Counter()
        self.release_year_counts: Counter[int] = Counter()
        self.release_decade_counts: Counter[str] = Counter()
        self.added_year_counts: Counter[int] = Counter()
        self.added_month_counts: Counter[str] = Counter()

        self.durations_minutes: List[float] = []
        self.duration_entries: List[Tuple[str, float]] = []
        self.ratings: List[float] = []
        self.rating_entries: List[Tuple[str, float]] = []
        self.actor_rating_totals: Dict[str, List[float]] = defaultdict(list)
        self.director_rating_totals: Dict[str, List[float]] = defaultdict(list)
        self.release_years: List[int] = []
        self.release_entries: List[Tuple[int, str]] = []
        self.addition_dates: List[datetime] = []
        self.addition_entries: List[Tuple[datetime, str]] = []
        self.view_entries: List[Tuple[str, int]] = []
        self.last_view_dates: List[datetime] = []
        self.metadata_gaps: Counter[str] = Counter()

        self.total_movies = 0
        self.total_view_count = 0
        self.epic_threshold_minutes = 150

    def add_movie(self, movie) -> None:
        self.total_movies += 1
        title = getattr(movie, "title", "Unknown Title")

        audience_rating = getattr(movie, "audienceRating", None)
        rating = audience_rating if isinstance(audience_rating, (int, float)) else getattr(movie, "rating", None)
        if isinstance(rating, (int, float)):
            rating_value = float(rating)
            self.ratings.append(rating_value)
            self.rating_entries.append((title, rating_value))

        year = getattr(movie, "year", None)
        if isinstance(year, str):
            year = int(year) if year.isdigit() else None
        if not year:
            original_date = getattr(movie, "originallyAvailableAt", None)
            if isinstance(original_date, datetime):
                year = original_date.year
        if isinstance(year, int):
            self.release_years.append(year)
            self.release_year_counts[year] += 1
            self.release_entries.append((year, title))
            decade = f"{(year // 10) * 10}s"
            self.release_decade_counts[decade] += 1

        added_at = getattr(movie, "addedAt", None)
        if isinstance(added_at, datetime):
            self.addition_dates.append(added_at)
            self.addition_entries.append((added_at, title))
            self.added_year_counts[added_at.year] += 1
            self.added_month_counts[added_at.strftime("%Y-%m")] += 1

        duration_ms = getattr(movie, "duration", None)
        if isinstance(duration_ms, (int, float)) and duration_ms > 0:
            duration_minutes = duration_ms / 60000
            self.durations_minutes.append(duration_minutes)
            self.duration_entries.append((title, duration_minutes))

        actors = getattr(movie, "actors", []) or []
        for actor in actors:
            if actor.tag:
                self.actor_counts[actor.tag] += 1
                if isinstance(rating, (int, float)):
                    self.actor_rating_totals[actor.tag].append(float(rating))

        directors = getattr(movie, "directors", []) or []
        for director in directors:
            if director.tag:
                self.director_counts[director.tag] += 1
                if isinstance(rating, (int, float)):
                    self.director_rating_totals[director.tag].append(float(rating))

        for genre in getattr(movie, "genres", []) or []:
            if genre.tag:
                self.genre_counts[genre.tag] += 1

        content_rating = getattr(movie, "contentRating", None)
        if content_rating:
            self.content_rating_counts[content_rating] += 1

        language = get_primary_audio_language(movie)
        if language:
            self.language_counts[language] += 1

        for country in getattr(movie, "countries", []) or []:
            tag = getattr(country, "tag", None)
            if tag:
                self.country_counts[tag] += 1

        for collection in getattr(movie, "collections", []) or []:
            tag = getattr(collection, "tag", None) or getattr(collection, "title", None)
            if tag:
                self.collection_counts[tag] += 1

        media_list = getattr(movie, "media", None) or []
        if media_list:
            primary_media = media_list[0]
            resolution = getattr(primary_media, "videoResolution", None)
            if resolution:
                self.resolution_counts[str(resolution).upper()] += 1

            dynamic_range = getattr(primary_media, "videoDynamicRange", None)
            if dynamic_range:
                self.hdr_counts[str(dynamic_range).upper()] += 1
            else:
                self.hdr_counts["SDR"] += 1

            audio_channels = getattr(primary_media, "audioChannels", None)
            if audio_channels:
                self.audio_channel_counts[str(audio_channels)] += 1

        guid_candidates = []
        guid_value = getattr(movie, "guid", None)
        if guid_value:
            guid_candidates.append(guid_value)
        for guid in getattr(movie, "guids", []) or []:
            candidate = getattr(guid, "id", None)
            if candidate:
                guid_candidates.append(candidate)
        for candidate in guid_candidates:
            provider = extract_provider_from_guid(candidate)
            if provider:
                self.source_counts[provider] += 1
                break

        view_count = getattr(movie, "viewCount", 0) or 0
        if isinstance(view_count, int) and view_count > 0:
            self.total_view_count += view_count
            self.view_entries.append((title, view_count))

        last_viewed = getattr(movie, "lastViewedAt", None)
        if isinstance(last_viewed, datetime):
            self.last_view_dates.append(last_viewed)

        summary = getattr(movie, "summary", "") or ""
        if not summary.strip():
            self.metadata_gaps["missing_summary"] += 1
        if not isinstance(rating, (int, float)):
            self.metadata_gaps["missing_rating"] += 1
        poster = getattr(movie, "thumb", None)
        if not poster:
            self.metadata_gaps["missing_poster"] += 1
        if not (getattr(movie, "genres", []) or []):
            self.metadata_gaps["missing_genre"] += 1

    def finalize(self) -> Dict[str, Any]:
        duration_stats = {
            "average_minutes": mean(self.durations_minutes) if self.durations_minutes else 0,
            "median_minutes": median(self.durations_minutes) if self.durations_minutes else 0,
            "shortest": min(self.duration_entries, key=lambda entry: entry[1]) if self.duration_entries else None,
            "longest": max(self.duration_entries, key=lambda entry: entry[1]) if self.duration_entries else None,
            "epic_count": sum(1 for _, minutes in self.duration_entries if minutes >= self.epic_threshold_minutes),
            "epic_threshold": self.epic_threshold_minutes,
        }

        rating_stats = {
            "average": mean(self.ratings) if self.ratings else None,
            "median": median(self.ratings) if self.ratings else None,
            "highest": sorted(self.rating_entries, key=lambda entry: entry[1], reverse=True)[:5],
            "lowest": sorted(self.rating_entries, key=lambda entry: entry[1])[:5],
            "threshold": 8.0,
            "above_threshold_count": sum(1 for _, value in self.rating_entries if value >= 8.0),
        }

        def compute_rankings(
            rating_map: Dict[str, List[float]],
            count_map: Counter[str],
            global_average: float | None,
            smoothing: int = 3,
        ) -> Dict[str, Dict[str, object]]:
            if not rating_map:
                return {
                    "highest": {"name": "n/a", "average": 0.0, "count": 0, "weighted": 0.0},
                    "lowest": {"name": "n/a", "average": 0.0, "count": 0, "weighted": 0.0},
                }

            prior = global_average if isinstance(global_average, (int, float)) else 0.0
            aggregates = []
            for name, values in rating_map.items():
                if not values:
                    continue
                count = count_map.get(name, len(values))
                average_rating = mean(values)
                weighted_rating = ((count / (count + smoothing)) * average_rating) + (
                    (smoothing / (count + smoothing)) * prior
                )
                aggregates.append(
                    {
                        "name": name,
                        "average": average_rating,
                        "count": count,
                        "weighted": weighted_rating,
                    }
                )

            if not aggregates:
                return {
                    "highest": {"name": "n/a", "average": 0.0, "count": 0, "weighted": 0.0},
                    "lowest": {"name": "n/a", "average": 0.0, "count": 0, "weighted": 0.0},
                }

            highest = max(aggregates, key=lambda entry: entry["weighted"])
            lowest = min(aggregates, key=lambda entry: entry["weighted"])
            return {"highest": highest, "lowest": lowest}

        person_rating_rankings = {
            "actor": compute_rankings(self.actor_rating_totals, self.actor_counts, rating_stats["average"]),
            "director": compute_rankings(self.director_rating_totals, self.director_counts, rating_stats["average"]),
        }

        year_stats = {
            "average": mean(self.release_years) if self.release_years else None,
            "median": median(self.release_years) if self.release_years else None,
            "earliest": min(self.release_entries, key=lambda entry: entry[0]) if self.release_entries else None,
            "latest": max(self.release_entries, key=lambda entry: entry[0]) if self.release_entries else None,
            "counts": self.release_year_counts,
            "decade_counts": self.release_decade_counts,
        }

        now = datetime.now()
        reference_now = now
        if self.addition_dates:
            reference_now = datetime.now(self.addition_dates[0].tzinfo) if self.addition_dates[0].tzinfo else now

        addition_stats = {
            "first_added": min(self.addition_entries, key=lambda entry: entry[0]) if self.addition_entries else None,
            "last_added": max(self.addition_entries, key=lambda entry: entry[0]) if self.addition_entries else None,
            "counts_by_year": self.added_year_counts,
            "counts_by_month": self.added_month_counts,
            "added_last_12_months": sum(
                1 for date in self.addition_dates if (reference_now - date) <= timedelta(days=365)
            ),
        }

        if addition_stats["last_added"]:
            last_added_date = addition_stats["last_added"][0]
            current = datetime.now(last_added_date.tzinfo) if last_added_date.tzinfo else now
            addition_stats["days_since_last_added"] = (current - last_added_date).days
        else:
            addition_stats["days_since_last_added"] = None

        watch_stats = {
            "total_plays": self.total_view_count,
            "movies_watched": len(self.view_entries),
            "most_watched": sorted(self.view_entries, key=lambda entry: entry[1], reverse=True)[:5],
            "last_viewed": max(self.last_view_dates) if self.last_view_dates else None,
        }

        return {
            "total_movies": self.total_movies,
            "actor_counts": self.actor_counts,
            "director_counts": self.director_counts,
            "genre_counts": self.genre_counts,
            "content_rating_counts": self.content_rating_counts,
            "language_counts": self.language_counts,
            "country_counts": self.country_counts,
            "collection_counts": self.collection_counts,
            "resolution_counts": self.resolution_counts,
            "hdr_counts": self.hdr_counts,
            "audio_channel_counts": self.audio_channel_counts,
            "source_counts": self.source_counts,
            "duration_stats": duration_stats,
            "rating_stats": rating_stats,
            "person_rating_rankings": person_rating_rankings,
            "year_stats": year_stats,
            "addition_stats": addition_stats,
            "watch_stats": watch_stats,
            "metadata_gaps": dict(self.metadata_gaps),
        }


def format_runtime_section(duration_stats: Dict[str, Any], total_movies: int) -> str:
    lines = []
    avg_minutes = duration_stats["average_minutes"]
    median_minutes = duration_stats["median_minutes"]
    lines.append(f"Average: {human_readable_duration(avg_minutes)} (~{avg_minutes:.0f} min)")
    lines.append(f"Median:  {human_readable_duration(median_minutes)} (~{median_minutes:.0f} min)")
    shortest = duration_stats["shortest"]
    if shortest:
        title, minutes = shortest
        lines.append(f"Shortest: {title} ({human_readable_duration(minutes)}, ~{minutes:.0f} min)")
    longest = duration_stats["longest"]
    if longest:
        title, minutes = longest
        lines.append(f"Longest: {title} ({human_readable_duration(minutes)}, ~{minutes:.0f} min)")
    epic_count = duration_stats["epic_count"]
    epic_percent = (epic_count / total_movies) * 100 if total_movies else 0
    lines.append(
        f"Epics (>={duration_stats['epic_threshold']} min): {epic_count} ({epic_percent:.1f}% of library)"
    )
    return "\n".join(lines)


def format_ratings_section(analysis: Dict[str, Any]) -> str:
    rating_stats = analysis["rating_stats"]
    person_rankings = analysis["person_rating_rankings"]
    lines = []
    avg_rating = rating_stats["average"]
    median_rating = rating_stats["median"]
    lines.append(f"Average: {avg_rating:.2f}" if avg_rating is not None else "Average: n/a")
    lines.append(f"Median: {median_rating:.2f}" if median_rating is not None else "Median: n/a")
    threshold_count = rating_stats["above_threshold_count"]
    threshold_percent = (
        (threshold_count / analysis["total_movies"]) * 100 if analysis["total_movies"] else 0
    )
    lines.append(
        f"Titles >={rating_stats['threshold']:.1f}: {threshold_count} ({threshold_percent:.1f}% of library)"
    )
    if rating_stats["highest"]:
        lines.append("Highest rated:")
        for title, value in rating_stats["highest"]:
            lines.append(f"  {title}: {value:.2f}")
    if rating_stats["lowest"]:
        lines.append("Lowest rated:")
        for title, value in rating_stats["lowest"]:
            lines.append(f"  {title}: {value:.2f}")

    def format_person_rank(label: str, data: Dict[str, Dict[str, object]]) -> List[str]:
        highest = data["highest"]
        lowest = data["lowest"]
        results = []
        if highest["name"] != "n/a":
            results.append(
                f"{label} highest avg rating: {highest['name']} ({highest['average']:.2f} avg over "
                f"{highest['count']} films, weighted {highest['weighted']:.2f})"
            )
        if lowest["name"] != "n/a":
            results.append(
                f"{label} lowest avg rating: {lowest['name']} ({lowest['average']:.2f} avg over "
                f"{lowest['count']} films, weighted {lowest['weighted']:.2f})"
            )
        return results

    actor_rank = person_rankings.get("actor")
    director_rank = person_rankings.get("director")
    if actor_rank:
        lines.extend(format_person_rank("Actor", actor_rank))
    if director_rank:
        lines.extend(format_person_rank("Director", director_rank))
    return "\n".join(lines)


def format_release_section(year_stats: Dict[str, Any]) -> str:
    lines = []
    average_year = year_stats["average"]
    median_year = year_stats["median"]
    lines.append(
        f"Average release year: {average_year:.0f}" if average_year is not None else "Average release year: n/a"
    )
    lines.append(
        f"Median release year: {median_year:.0f}" if median_year is not None else "Median release year: n/a"
    )
    if year_stats["earliest"]:
        earliest_year, earliest_title = year_stats["earliest"]
        lines.append(f"Oldest title: {earliest_title} ({earliest_year})")
    if year_stats["latest"]:
        latest_year, latest_title = year_stats["latest"]
        lines.append(f"Newest title: {latest_title} ({latest_year})")
    top_decades = year_stats["decade_counts"].most_common(5)
    if top_decades:
        lines.append("Favorite decades:")
        for decade, count in top_decades:
            lines.append(f"  {decade}: {count}")
    top_years = year_stats["counts"].most_common(5)
    if top_years:
        lines.append("Busiest release years:")
        for year, count in top_years:
            lines.append(f"  {year}: {count}")
    return "\n".join(lines)


def format_growth_section(addition_stats: Dict[str, Any]) -> str:
    lines = []
    first_added = addition_stats["first_added"]
    if first_added:
        first_date, first_title = first_added
        lines.append(f"First added: {format_datetime_short(first_date)} - {first_title}")
    last_added = addition_stats["last_added"]
    if last_added:
        last_date, last_title = last_added
        lines.append(f"Most recent addition: {format_datetime_short(last_date)} - {last_title}")
    if addition_stats["days_since_last_added"] is not None:
        lines.append(f"Days since last addition: {addition_stats['days_since_last_added']}")
    lines.append(f"Added in last 12 months: {addition_stats['added_last_12_months']}")
    growth_years = addition_stats["counts_by_year"].most_common(5)
    if growth_years:
        lines.append("Peak add years:")
        for year, count in growth_years:
            lines.append(f"  {year}: {count}")
    growth_months = sorted(
        addition_stats["counts_by_month"].items(),
        key=lambda item: item[0],
        reverse=True,
    )[:6]
    if growth_months:
        lines.append("Recent additions by month:")
        for month, count in growth_months:
            lines.append(f"  {month}: {count}")
    return "\n".join(lines)


def format_watch_section(watch_stats: Dict[str, Any], total_movies: int) -> str:
    lines = []
    lines.append(f"Total plays tracked: {watch_stats['total_plays']}")
    watched_percent = (watch_stats["movies_watched"] / total_movies * 100) if total_movies else 0
    lines.append(
        f"Titles watched: {watch_stats['movies_watched']} ({watched_percent:.1f}% of library)"
    )
    if watch_stats["last_viewed"]:
        lines.append(f"Last watched: {format_datetime_short(watch_stats['last_viewed'])}")
    if watch_stats["most_watched"]:
        lines.append("Most rewatched:")
        for title, plays in watch_stats["most_watched"]:
            lines.append(f"  {title}: {plays} plays")
    return "\n".join(lines)


def format_datetime_short(value: datetime | None) -> str:
    if not value:
        return "n/a"
    if value.tzinfo:
        value = value.astimezone()
    return value.strftime("%Y-%m-%d")


def format_sections(analysis: Dict[str, Any]) -> Dict[str, str]:
    sections = {}
    sections["runtime"] = format_runtime_section(analysis["duration_stats"], analysis["total_movies"])
    sections["ratings"] = format_ratings_section(analysis)
    sections["release"] = format_release_section(analysis["year_stats"])
    sections["growth"] = format_growth_section(analysis["addition_stats"])
    sections["watch"] = format_watch_section(analysis["watch_stats"], analysis["total_movies"])
    sections["talent"] = "\n".join(
        [
            "Top actors:",
            format_counter(analysis["actor_counts"]),
            "",
            "Top directors:",
            format_counter(analysis["director_counts"]),
            "",
            "Top genres:",
            format_counter(analysis["genre_counts"]),
        ]
    )
    sections["audience"] = "\n".join(
        [
            "Content ratings:",
            format_counter(analysis["content_rating_counts"]),
            "",
            "Audio languages:",
            format_counter(analysis["language_counts"]),
            "",
            "Countries of origin:",
            format_counter(analysis["country_counts"]),
        ]
    )
    sections["collections"] = "\n".join(
        [
            "Collections:",
            format_counter(analysis["collection_counts"]),
            "",
            "Metadata sources:",
            format_counter(analysis["source_counts"], transform=pretty_provider),
        ]
    )
    sections["technical"] = "\n".join(
        [
            "Video resolutions:",
            format_counter(analysis["resolution_counts"]),
            "",
            "Dynamic ranges:",
            format_counter(analysis["hdr_counts"]),
            "",
            "Audio channel layouts:",
            format_counter(analysis["audio_channel_counts"]),
        ]
    )
    sections["metadata"] = format_metadata_gaps(analysis["metadata_gaps"])
    return sections


SECTION_LAYOUT = [
    ("runtime", "Runtime Profile", (60, 6)),
    ("ratings", "Ratings", (60, 12)),
    ("release", "Release Timeline", (60, 9)),
    ("growth", "Library Growth", (60, 9)),
    ("watch", "Watch Activity", (60, 7)),
    ("talent", "Talent & Genres", (60, 11)),
    ("audience", "Audience & Origins", (60, 11)),
    ("collections", "Collections & Sources", (60, 10)),
    ("technical", "Technical Snapshot", (60, 11)),
    ("metadata", "Metadata Gaps", (60, 7)),
]


def build_window() -> sg.Window:
    layout = [
        [sg.Text("Status:", size=(8, 1)), sg.Text("", key="status", size=(80, 1))],
    ]
    for key, title, size in SECTION_LAYOUT:
        layout.append(
            [
                sg.Frame(
                    title,
                    [[sg.Multiline("", size=size, key=key, disabled=True, autoscroll=True)]],
                    expand_x=True,
                )
            ]
        )
    layout.append([sg.Button("Exit")])
    return sg.Window("Plex Movie Analytics", layout, finalize=True, resizable=True)


def analysis_worker(env: Dict[str, str], message_queue: Queue, stop_event: threading.Event) -> None:
    try:
        message_queue.put(("status", "Connecting to Plex..."))
        plex = connect_to_plex(env["PLEX_URL"], env["PLEX_TOKEN"])
        library_name = env["PLEX_LIBRARY_NAME"]
        try:
            library = plex.library.section(library_name)
        except NotFound as err:
            raise RuntimeError(
                f"Plex library '{library_name}' was not found."
            ) from err

        message_queue.put(("status", f"Fetching movies from '{library.title}'..."))
        movies = list(fetch_movies(library))
        total_movies = len(movies)
        message_queue.put(("status", f"Processing {total_movies} movies..."))

        accumulator = AnalysisAccumulator()
        for index, movie in enumerate(movies, start=1):
            if stop_event.is_set():
                message_queue.put(("status", "Analysis cancelled by user."))
                return

            accumulator.add_movie(movie)
            if index % 10 == 0 or index == total_movies:
                message_queue.put(
                    ("status", f"Processing {index}/{total_movies}: {getattr(movie, 'title', 'Unknown')}")
                )

        analysis = accumulator.finalize()
        sections = format_sections(analysis)
        for key, text in sections.items():
            message_queue.put(("section", key, text))

        message_queue.put(
            ("status", f"Analysis complete. Processed {analysis['total_movies']} movies.")
        )
    except Exception as exc:  # pragma: no cover - GUI entry point
        logger.exception("Analysis failed")
        message_queue.put(("error", str(exc)))


def drain_queue(window: sg.Window, message_queue: Queue) -> None:
    while True:
        try:
            message = message_queue.get_nowait()
        except Empty:
            break

        message_type = message[0]
        if message_type == "status":
            status_text = message[1]
            logger.info(status_text)
            window["status"].update(status_text)
        elif message_type == "section":
            _, key, text = message
            if key in window.AllKeysDict:
                window[key].update(text)
        elif message_type == "error":
            error_text = message[1]
            logger.error("Error from worker: %s", error_text)
            window["status"].update(f"Error: {error_text}")
        else:
            logger.debug("Unknown message type received: %s", message_type)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging(args.log_level)
    sg.theme(args.theme)

    load_env_file(ENV_FILE)
    try:
        env = get_required_env(REQUIRED_ENV_KEYS)
    except Exception as exc:
        logger.error("Failed to load environment configuration: %s", exc)
        sys.exit(1)

    window = build_window()
    message_queue: Queue = Queue()
    stop_event = threading.Event()

    worker_thread = threading.Thread(
        target=analysis_worker,
        args=(env, message_queue, stop_event),
        daemon=True,
    )
    worker_thread.start()

    try:
        while True:
            event, _ = window.read(timeout=200)
            if event in (sg.WIN_CLOSED, "Exit"):
                stop_event.set()
                break
            drain_queue(window, message_queue)
    finally:
        stop_event.set()
        worker_thread.join(timeout=5)
        window.close()


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except Exception as exc:  # pragma: no cover
        sys.stderr.write(f"Error: {exc}\n")
        sys.exit(1)



import os
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List

try:
    from plexapi.exceptions import NotFound, Unauthorized
    from plexapi.server import PlexServer
except ImportError as exc:  # pragma: no cover - dependency hint
    sys.stderr.write(
        "Missing dependency: plexapi.\n"
        "Install it with 'pip install plexapi'.\n"
    )
    raise


ENV_FILE = Path(__file__).with_name(".env")
REQUIRED_ENV_KEYS = ("PLEX_URL", "PLEX_TOKEN", "PLEX_LIBRARY_NAME")


def load_env_file(env_path: Path) -> None:
    """Populate os.environ with values from a simple KEY=VALUE .env file."""
    if not env_path.exists():
        raise FileNotFoundError(f"Environment file not found: {env_path}")

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')

        if key and key not in os.environ:
            os.environ[key] = value


def get_required_env(keys: Iterable[str]) -> Dict[str, str]:
    missing = [key for key in keys if not os.environ.get(key)]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(f"Missing required environment variables: {joined}")
    return {key: os.environ[key] for key in keys}


def connect_to_plex(url: str, token: str) -> PlexServer:
    try:
        return PlexServer(url, token)
    except Unauthorized as err:
        raise RuntimeError("Plex authorization failed. Check PLEX_TOKEN.") from err
    except NotFound as err:
        raise RuntimeError("Plex server could not be reached. Check PLEX_URL.") from err


def fetch_movies(library) -> Iterable:
    try:
        # Fetching all items can take time; PlexAPI handles paging internally.
        return library.all(libtype="movie")
    except NotFound as err:
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


def analyze_movies(movies: Iterable) -> Dict[str, object]:
    actor_counts: Counter[str] = Counter()
    director_counts: Counter[str] = Counter()
    genre_counts: Counter[str] = Counter()
    content_rating_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    country_counts: Counter[str] = Counter()
    collection_counts: Counter[str] = Counter()
    resolution_counts: Counter[str] = Counter()
    hdr_counts: Counter[str] = Counter()
    audio_channel_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    release_year_counts: Counter[int] = Counter()
    release_decade_counts: Counter[str] = Counter()
    added_year_counts: Counter[int] = Counter()
    added_month_counts: Counter[str] = Counter()

    durations_minutes = []
    duration_entries = []
    ratings = []
    rating_entries = []
    actor_rating_totals: Dict[str, List[float]] = {}
    director_rating_totals: Dict[str, List[float]] = {}
    release_years = []
    release_entries = []
    addition_dates = []
    addition_entries = []
    view_entries = []
    last_view_dates = []
    metadata_gaps: Counter[str] = Counter()

    total_movies = 0
    total_view_count = 0
    epic_threshold_minutes = 150

    for movie in movies:
        total_movies += 1
        title = getattr(movie, "title", "Unknown Title")

        # Release year information
        year = getattr(movie, "year", None)
        if isinstance(year, str):
            year = int(year) if year.isdigit() else None
        if not year:
            original_date = getattr(movie, "originallyAvailableAt", None)
            if isinstance(original_date, datetime):
                year = original_date.year
        if isinstance(year, int):
            release_years.append(year)
            release_year_counts[year] += 1
            release_entries.append((year, title))
            decade = f"{(year // 10) * 10}s"
            release_decade_counts[decade] += 1

        # Library addition timeline
        added_at = getattr(movie, "addedAt", None)
        if isinstance(added_at, datetime):
            addition_dates.append(added_at)
            addition_entries.append((added_at, title))
            added_year_counts[added_at.year] += 1
            added_month_counts[added_at.strftime("%Y-%m")] += 1

        # Duration statistics (convert from milliseconds to minutes)
        duration_ms = getattr(movie, "duration", None)
        if isinstance(duration_ms, (int, float)) and duration_ms > 0:
            duration_minutes = duration_ms / 60000
            durations_minutes.append(duration_minutes)
            duration_entries.append((title, duration_minutes))

        # Rating statistics
        rating = getattr(movie, "rating", None)
        if isinstance(rating, (int, float)):
            rating_value = float(rating)
            ratings.append(rating_value)
            rating_entries.append((title, rating_value))

        # Cast & crew breakdowns
        actors = getattr(movie, "actors", []) or []
        for actor in actors:
            if actor.tag:
                actor_counts[actor.tag] += 1
                if isinstance(rating, (int, float)):
                    actor_rating_totals.setdefault(actor.tag, []).append(float(rating))

        directors = getattr(movie, "directors", []) or []
        for director in directors:
            if director.tag:
                director_counts[director.tag] += 1
                if isinstance(rating, (int, float)):
                    director_rating_totals.setdefault(director.tag, []).append(float(rating))

        for genre in getattr(movie, "genres", []) or []:
            if genre.tag:
                genre_counts[genre.tag] += 1

        # Content ratings
        content_rating = getattr(movie, "contentRating", None)
        if content_rating:
            content_rating_counts[content_rating] += 1

        # Localization
        language = get_primary_audio_language(movie)
        if language:
            language_counts[language] += 1

        for country in getattr(movie, "countries", []) or []:
            tag = getattr(country, "tag", None)
            if tag:
                country_counts[tag] += 1

        # Collections / franchises
        for collection in getattr(movie, "collections", []) or []:
            tag = getattr(collection, "tag", None) or getattr(collection, "title", None)
            if tag:
                collection_counts[tag] += 1

        # Technical stats
        media_list = getattr(movie, "media", None) or []
        if media_list:
            primary_media = media_list[0]
            resolution = getattr(primary_media, "videoResolution", None)
            if resolution:
                resolution_counts[str(resolution).upper()] += 1

            dynamic_range = getattr(primary_media, "videoDynamicRange", None)
            if dynamic_range:
                hdr_counts[str(dynamic_range).upper()] += 1
            else:
                hdr_counts["SDR"] += 1

            audio_channels = getattr(primary_media, "audioChannels", None)
            if audio_channels:
                audio_channel_counts[str(audio_channels)] += 1

        # Streaming source
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
                source_counts[provider] += 1
                break  # Count the first recognizable provider

        # Watch history
        view_count = getattr(movie, "viewCount", 0) or 0
        if isinstance(view_count, int) and view_count > 0:
            total_view_count += view_count
            view_entries.append((title, view_count))

        last_viewed = getattr(movie, "lastViewedAt", None)
        if isinstance(last_viewed, datetime):
            last_view_dates.append(last_viewed)

        # Metadata completeness
        summary = getattr(movie, "summary", "") or ""
        if not summary.strip():
            metadata_gaps["missing_summary"] += 1

        if not isinstance(rating, (int, float)):
            metadata_gaps["missing_rating"] += 1

        poster = getattr(movie, "thumb", None)
        if not poster:
            metadata_gaps["missing_poster"] += 1

        if not (getattr(movie, "genres", []) or []):
            metadata_gaps["missing_genre"] += 1

    duration_stats = {
        "average_minutes": mean(durations_minutes) if durations_minutes else 0,
        "median_minutes": median(durations_minutes) if durations_minutes else 0,
        "shortest": min(duration_entries, key=lambda entry: entry[1]) if duration_entries else None,
        "longest": max(duration_entries, key=lambda entry: entry[1]) if duration_entries else None,
        "epic_count": sum(1 for _, minutes in duration_entries if minutes >= epic_threshold_minutes),
        "epic_threshold": epic_threshold_minutes,
    }

    rating_stats = {
        "average": mean(ratings) if ratings else None,
        "median": median(ratings) if ratings else None,
        "highest": sorted(rating_entries, key=lambda entry: entry[1], reverse=True)[:5],
        "lowest": sorted(rating_entries, key=lambda entry: entry[1])[:5],
        "threshold": 8.0,
        "above_threshold_count": sum(1 for _, value in rating_entries if value >= 8.0),
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
        "actor": compute_rankings(actor_rating_totals, actor_counts, rating_stats["average"]),
        "director": compute_rankings(director_rating_totals, director_counts, rating_stats["average"]),
    }

    year_stats = {
        "average": mean(release_years) if release_years else None,
        "median": median(release_years) if release_years else None,
        "earliest": min(release_entries, key=lambda entry: entry[0]) if release_entries else None,
        "latest": max(release_entries, key=lambda entry: entry[0]) if release_entries else None,
        "counts": release_year_counts,
        "decade_counts": release_decade_counts,
    }

    now = datetime.now()
    if addition_dates:
        reference_now = datetime.now(addition_dates[0].tzinfo) if addition_dates[0].tzinfo else now
    else:
        reference_now = now

    addition_stats = {
        "first_added": min(addition_entries, key=lambda entry: entry[0]) if addition_entries else None,
        "last_added": max(addition_entries, key=lambda entry: entry[0]) if addition_entries else None,
        "counts_by_year": added_year_counts,
        "counts_by_month": added_month_counts,
        "added_last_12_months": sum(
            1 for date in addition_dates if (reference_now - date) <= timedelta(days=365)
        ),
    }

    if addition_stats["last_added"]:
        last_added_date = addition_stats["last_added"][0]
        if last_added_date.tzinfo:
            current = datetime.now(last_added_date.tzinfo)
        else:
            current = now
        addition_stats["days_since_last_added"] = (current - last_added_date).days
    else:
        addition_stats["days_since_last_added"] = None

    watch_stats = {
        "total_plays": total_view_count,
        "movies_watched": len(view_entries),
        "most_watched": sorted(view_entries, key=lambda entry: entry[1], reverse=True)[:5],
        "last_viewed": max(last_view_dates) if last_view_dates else None,
    }

    return {
        "total_movies": total_movies,
        "actor_counts": actor_counts,
        "director_counts": director_counts,
        "genre_counts": genre_counts,
        "content_rating_counts": content_rating_counts,
        "language_counts": language_counts,
        "country_counts": country_counts,
        "collection_counts": collection_counts,
        "resolution_counts": resolution_counts,
        "hdr_counts": hdr_counts,
        "audio_channel_counts": audio_channel_counts,
        "source_counts": source_counts,
        "duration_stats": duration_stats,
        "rating_stats": rating_stats,
        "person_rating_rankings": person_rating_rankings,
        "year_stats": year_stats,
        "addition_stats": addition_stats,
        "watch_stats": watch_stats,
        "metadata_gaps": dict(metadata_gaps),
    }


def human_readable_duration(minutes_value: float) -> str:
    if not minutes_value:
        return "n/a"
    total_minutes = int(round(minutes_value))
    hours, minutes = divmod(total_minutes, 60)
    return f"{hours}h {minutes}m" if hours else f"{minutes}m"


def print_top(
    counter: Counter,
    label: str,
    limit: int = 5,
    transform=lambda value: value,
    indent: str = "",
    show_empty: bool = False,
) -> None:
    if not counter:
        if show_empty:
            print(f"{indent}No {label} data available.")
        return

    print(f"{indent}Top {label}:")
    for name, count in counter.most_common(limit):
        print(f"{indent}  {transform(name)}: {count}")


def format_datetime_short(value: datetime | None) -> str:
    if not value:
        return "n/a"
    if value.tzinfo:
        value = value.astimezone()
    return value.strftime("%Y-%m-%d")


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


def main() -> None:
    try:
        load_env_file(ENV_FILE)
        env = get_required_env(REQUIRED_ENV_KEYS)

        plex = connect_to_plex(env["PLEX_URL"], env["PLEX_TOKEN"])
        try:
            library = plex.library.section(env["PLEX_LIBRARY_NAME"])
        except NotFound as err:
            raise RuntimeError(
                f"Plex library '{env['PLEX_LIBRARY_NAME']}' was not found."
            ) from err

        movies = list(fetch_movies(library))
        analysis = analyze_movies(movies)

        print(f"Analyzed {analysis['total_movies']} movies from '{library.title}'.\n")

        duration_stats = analysis["duration_stats"]
        print("Runtime Profile")
        avg_minutes = duration_stats["average_minutes"]
        median_minutes = duration_stats["median_minutes"]
        print(
            f"  Average: {human_readable_duration(avg_minutes)} (~{avg_minutes:.0f} min)"
        )
        print(
            f"  Median: {human_readable_duration(median_minutes)} (~{median_minutes:.0f} min)"
        )
        shortest = duration_stats["shortest"]
        if shortest:
            short_title, short_minutes = shortest
            print(
                f"  Shortest: {short_title} ({human_readable_duration(short_minutes)}"
                f", ~{short_minutes:.0f} min)"
            )
        longest = duration_stats["longest"]
        if longest:
            long_title, long_minutes = longest
            print(
                f"  Longest: {long_title} ({human_readable_duration(long_minutes)}"
                f", ~{long_minutes:.0f} min)"
            )
        epic_count = duration_stats["epic_count"]
        epic_percent = (
            (epic_count / analysis["total_movies"]) * 100 if analysis["total_movies"] else 0
        )
        print(
            f"  Epics (>={duration_stats['epic_threshold']} min): {epic_count}"
            f" ({epic_percent:.1f}% of library)"
        )

        rating_stats = analysis["rating_stats"]
        print("\nRatings")
        avg_rating = rating_stats["average"]
        if avg_rating is not None:
            print(f"  Average: {avg_rating:.2f}")
        else:
            print("  Average: n/a")
        median_rating = rating_stats["median"]
        if median_rating is not None:
            print(f"  Median: {median_rating:.2f}")
        else:
            print("  Median: n/a")
        threshold_count = rating_stats["above_threshold_count"]
        threshold_percent = (
            (threshold_count / analysis["total_movies"]) * 100
            if analysis["total_movies"]
            else 0
        )
        print(
            f"  Titles >={rating_stats['threshold']:.1f}: {threshold_count}"
            f" ({threshold_percent:.1f}% of library)"
        )
        if rating_stats["highest"]:
            print("  Highest rated:")
            for title, value in rating_stats["highest"]:
                print(f"    {title}: {value:.2f}")
        if rating_stats["lowest"]:
            print("  Lowest rated:")
            for title, value in rating_stats["lowest"]:
                print(f"    {title}: {value:.2f}")
        person_rating_rankings = analysis.get("person_rating_rankings", {})
        actor_rank = person_rating_rankings.get("actor")
        director_rank = person_rating_rankings.get("director")
        if actor_rank:
            highest_actor = actor_rank["highest"]
            lowest_actor = actor_rank["lowest"]
            if highest_actor["name"] != "n/a":
                print(
                    "  Actor highest avg rating: "
                    f"{highest_actor['name']} ({highest_actor['average']:.2f} avg over "
                    f"{highest_actor['count']} films, weighted {highest_actor['weighted']:.2f})"
                )
            if lowest_actor["name"] != "n/a":
                print(
                    "  Actor lowest avg rating: "
                    f"{lowest_actor['name']} ({lowest_actor['average']:.2f} avg over "
                    f"{lowest_actor['count']} films, weighted {lowest_actor['weighted']:.2f})"
                )
        if director_rank:
            highest_director = director_rank["highest"]
            lowest_director = director_rank["lowest"]
            if highest_director["name"] != "n/a":
                print(
                    "  Director highest avg rating: "
                    f"{highest_director['name']} ({highest_director['average']:.2f} avg over "
                    f"{highest_director['count']} films, weighted {highest_director['weighted']:.2f})"
                )
            if lowest_director["name"] != "n/a":
                print(
                    "  Director lowest avg rating: "
                    f"{lowest_director['name']} ({lowest_director['average']:.2f} avg over "
                    f"{lowest_director['count']} films, weighted {lowest_director['weighted']:.2f})"
                )

        year_stats = analysis["year_stats"]
        print("\nRelease Timeline")
        average_year = year_stats["average"]
        if average_year is not None:
            print(f"  Average release year: {average_year:.0f}")
        median_year = year_stats["median"]
        if median_year is not None:
            print(f"  Median release year: {median_year:.0f}")
        if year_stats["earliest"]:
            earliest_year, earliest_title = year_stats["earliest"]
            print(f"  Oldest title: {earliest_title} ({earliest_year})")
        if year_stats["latest"]:
            latest_year, latest_title = year_stats["latest"]
            print(f"  Newest title: {latest_title} ({latest_year})")
        top_decades = year_stats["decade_counts"].most_common(5)
        if top_decades:
            print("  Favorite decades:")
            for decade, count in top_decades:
                print(f"    {decade}: {count}")
        top_years = year_stats["counts"].most_common(5)
        if top_years:
            print("  Busiest years of release:")
            for year, count in top_years:
                print(f"    {year}: {count}")

        addition_stats = analysis["addition_stats"]
        print("\nLibrary Growth")
        first_added = addition_stats["first_added"]
        if first_added:
            first_date, first_title = first_added
            print(f"  First added: {format_datetime_short(first_date)} - {first_title}")
        last_added = addition_stats["last_added"]
        if last_added:
            last_date, last_title = last_added
            print(f"  Most recent addition: {format_datetime_short(last_date)} - {last_title}")
        if addition_stats["days_since_last_added"] is not None:
            print(
                f"  Days since last addition: {addition_stats['days_since_last_added']}"
            )
        print(f"  Added in last 12 months: {addition_stats['added_last_12_months']}")
        growth_years = addition_stats["counts_by_year"].most_common(5)
        if growth_years:
            print("  Peak add years:")
            for year, count in growth_years:
                print(f"    {year}: {count}")
        growth_months = sorted(
            addition_stats["counts_by_month"].items(),
            key=lambda item: item[0],
            reverse=True,
        )[:6]
        if growth_months:
            print("  Recent additions by month:")
            for month, count in growth_months:
                print(f"    {month}: {count}")

        watch_stats = analysis["watch_stats"]
        print("\nWatch Activity")
        print(f"  Total plays tracked: {watch_stats['total_plays']}")
        watched_percent = (
            (watch_stats["movies_watched"] / analysis["total_movies"]) * 100
            if analysis["total_movies"]
            else 0
        )
        print(
            f"  Titles watched: {watch_stats['movies_watched']}"
            f" ({watched_percent:.1f}% of library)"
        )
        if watch_stats["last_viewed"]:
            print(f"  Last watched: {format_datetime_short(watch_stats['last_viewed'])}")
        if watch_stats["most_watched"]:
            print("  Most rewatched:")
            for title, plays in watch_stats["most_watched"]:
                print(f"    {title}: {plays} plays")

        print("\nTalent & Genres")
        print_top(analysis["actor_counts"], "actors", indent="  ")
        print_top(analysis["director_counts"], "directors", indent="  ")
        print_top(analysis["genre_counts"], "genres", indent="  ")

        print("\nAudience & Origins")
        print_top(analysis["content_rating_counts"], "content ratings", indent="  ")
        print_top(analysis["language_counts"], "audio languages", indent="  ")
        print_top(analysis["country_counts"], "countries of origin", indent="  ")

        print("\nCollections & Sources")
        print_top(analysis["collection_counts"], "collections", indent="  ")
        print_top(
            analysis["source_counts"],
            "metadata sources",
            indent="  ",
            transform=pretty_provider,
        )

        print("\nTechnical Snapshot")
        print_top(analysis["resolution_counts"], "video resolutions", indent="  ")
        print_top(analysis["hdr_counts"], "dynamic ranges", indent="  ")
        print_top(analysis["audio_channel_counts"], "audio channel layouts", indent="  ")

        print("\nMetadata Gaps")
        metadata_gaps = {
            key: value for key, value in analysis["metadata_gaps"].items() if value
        }
        if metadata_gaps:
            for key, value in sorted(
                metadata_gaps.items(), key=lambda item: item[1], reverse=True
            ):
                friendly = key.replace("_", " ").title()
                print(f"  {friendly}: {value}")
        else:
            print("  None")

    except Exception as exc:
        sys.stderr.write(f"Error: {exc}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()

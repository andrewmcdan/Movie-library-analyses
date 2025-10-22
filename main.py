from __future__ import annotations

import os
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable

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


def analyze_movies(movies: Iterable) -> Dict[str, object]:
    actor_counts: Counter[str] = Counter()
    director_counts: Counter[str] = Counter()
    genre_counts: Counter[str] = Counter()
    durations = []
    ratings = []
    total_movies = 0

    for movie in movies:
        print(f"Analyzing movie: {getattr(movie, 'title', 'n/a')}")
        total_movies += 1

        if getattr(movie, "duration", None):
            durations.append(movie.duration)  # duration is in milliseconds

        rating = getattr(movie, "rating", None)
        if isinstance(rating, (int, float)):
            ratings.append(float(rating))

        for actor in getattr(movie, "actors", []) or []:
            if actor.tag:
                actor_counts[actor.tag] += 1

        for director in getattr(movie, "directors", []) or []:
            if director.tag:
                director_counts[director.tag] += 1

        for genre in getattr(movie, "genres", []) or []:
            if genre.tag:
                genre_counts[genre.tag] += 1

    average_duration = mean(durations) if durations else 0
    average_rating = mean(ratings) if ratings else None

    return {
        "total_movies": total_movies,
        "actor_counts": actor_counts,
        "director_counts": director_counts,
        "genre_counts": genre_counts,
        "average_duration": average_duration,
        "average_rating": average_rating,
    }


def human_readable_duration(minutes_value: float) -> str:
    if not minutes_value:
        return "n/a"
    total_minutes = int(round(minutes_value))
    hours, minutes = divmod(total_minutes, 60)
    return f"{hours}h {minutes}m" if hours else f"{minutes}m"


def print_top(counter: Counter[str], label: str, limit: int = 5) -> None:
    if not counter:
        print(f"No {label.lower()} data available.")
        return

    print(f"Top {label}:")
    for name, count in counter.most_common(limit):
        print(f"  {name}: {count}")


def main() -> None:
    print("Starting Plex movie library analysis...\n")
    try:
        load_env_file(ENV_FILE)
        env = get_required_env(REQUIRED_ENV_KEYS)

        plex = connect_to_plex(env["PLEX_URL"], env["PLEX_TOKEN"])
        print("Connected to Plex server.\n")
        try:
            library = plex.library.section(env["PLEX_LIBRARY_NAME"])
        except NotFound as err:
            raise RuntimeError(
                f"Plex library '{env['PLEX_LIBRARY_NAME']}' was not found."
            ) from err

        movies = list(fetch_movies(library))
        print(f"Fetched {len(movies)} movies from library '{library.title}'.\n")
        analysis = analyze_movies(movies)

        print(f"Analyzed {analysis['total_movies']} movies from '{library.title}'.")
        print(f"Average runtime: {human_readable_duration(analysis['average_duration'])}")
        avg_rating = analysis['average_rating']
        if avg_rating is not None:
            print(f"Average community rating: {avg_rating:.2f}")
        else:
            print("Average community rating: n/a")

        print()
        print_top(analysis["actor_counts"], "actors")
        print()
        print_top(analysis["director_counts"], "directors")
        print()
        print_top(analysis["genre_counts"], "genres")

    except Exception as exc:
        sys.stderr.write(f"Error: {exc}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()

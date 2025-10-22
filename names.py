from __future__ import annotations

import json
import os
import re
import sys
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Any

import logging

try:
    from plexapi.exceptions import NotFound, Unauthorized
    from plexapi.server import PlexServer
except ImportError as exc:  # pragma: no cover - dependency hint
    sys.stderr.write(
        "Missing dependency: plexapi.\n"
        "Install it with 'pip install plexapi'.\n"
    )
    raise

try:
    from openai import OpenAI  # type: ignore
except ImportError:
    OpenAI = None  # type: ignore

try:
    import openai  # type: ignore
except ImportError:
    openai = None  # type: ignore


TRACE_LEVEL_NUM = 5
LOG_LEVELS = {
    "trace": TRACE_LEVEL_NUM,
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warn": logging.WARNING,
    "error": logging.ERROR,
}


def _configure_trace_level() -> None:
    if hasattr(logging, "TRACE"):
        return
    logging.addLevelName(TRACE_LEVEL_NUM, "TRACE")

    def trace(self, message, *args, **kwargs):
        if self.isEnabledFor(TRACE_LEVEL_NUM):
            self._log(TRACE_LEVEL_NUM, message, args, **kwargs)

    setattr(logging.Logger, "trace", trace)  # type: ignore[attr-defined]


_configure_trace_level()
logger = logging.getLogger("names")


@dataclass
class OpenAIChatClient:
    model: str
    temperature: float = 1.0

    def __post_init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to your environment or .env file."
            )

        if OpenAI is not None:
            logger.debug("Initializing modern OpenAI client.")
            self._mode = "modern"
            self._client = OpenAI(api_key=api_key)
            return

        if openai is not None:
            logger.debug("Initializing legacy OpenAI client.")
            openai.api_key = api_key
            self._mode = "legacy"
            self._client = openai
            return

        raise RuntimeError(
            "OpenAI SDK not found. Install it with 'pip install openai'."
        )

    def complete(self, messages: List[Dict[str, str]]) -> str:
        logger.debug(
            "Requesting OpenAI completion (model=%s, temperature=%s, messages=%d)",
            self.model,
            self.temperature,
            len(messages),
        )
        try:
            if self._mode == "modern":
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                )
                content = response.choices[0].message.content or ""
            else:
                response = self._client.ChatCompletion.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                )
                content = response["choices"][0]["message"]["content"] or ""
            logger.debug("OpenAI response length: %d characters", len(content))
            return content
        except Exception as exc:  # pragma: no cover - API errors
            logger.exception("OpenAI chat request failed.")
            raise RuntimeError(f"OpenAI chat request failed: {exc}") from exc


ENV_FILE = Path(__file__).with_name(".env")
REQUIRED_ENV_KEYS = ("PLEX_URL", "PLEX_TOKEN", "PLEX_LIBRARY_NAME", "OPENAI_API_KEY")


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        raise FileNotFoundError(f"Environment file not found: {env_path}")

    logger.debug("Loading environment variables from %s", env_path)
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')

        if key and key not in os.environ:
            os.environ[key] = value
            logger.trace("Loaded %s from env file", key)


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
    logger.debug("Logging configured for level: %s", level_name.upper())


def parse_args(argv: List[str] | None) -> Namespace:
    parser = ArgumentParser(description="Validate Plex movie filenames against metadata.")
    parser.add_argument(
        "--log-level",
        choices=LOG_LEVELS.keys(),
        default="info",
        help="Logging level to use (default: info).",
    )
    parser.add_argument(
        "--cache-file",
        default="metadata_matches.json",
        help="Path to JSON cache for confirmed matches (default: metadata_matches.json).",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Ignore existing cache entries and revalidate all files.",
    )
    return parser.parse_args(argv)


def get_required_env(keys: Iterable[str]) -> Dict[str, str]:
    missing = [key for key in keys if not os.environ.get(key)]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(f"Missing required environment variables: {joined}")
    env = {key: os.environ[key] for key in keys}
    logger.debug("Environment variables loaded: %s", ", ".join(env.keys()))
    return env


def load_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        logger.debug("Cache file %s not found; starting fresh.", path)
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            logger.info("Loaded %d cached confirmation(s) from %s.", len(data), path)
            return data
        logger.warning("Cache file %s did not contain a JSON object. Resetting.", path)
    except Exception as exc:
        logger.warning("Failed to read cache file %s: %s. Resetting cache.", path, exc)
    return {}


def save_cache(path: Path, cache: Dict[str, Dict[str, Any]]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")
        logger.info("Persisted %d cached confirmation(s) to %s.", len(cache), path)
    except Exception as exc:
        logger.error("Failed to write cache file %s: %s", path, exc)


def connect_to_plex(url: str, token: str) -> PlexServer:
    try:
        logger.info("Connecting to Plex at %s", url)
        server = PlexServer(url, token)
        logger.debug("Successfully connected to Plex server: %s", server.friendlyName)
        return server
    except Unauthorized as err:
        logger.error("Plex authorization failed: %s", err)
        raise RuntimeError("Plex authorization failed. Check PLEX_TOKEN.") from err
    except NotFound as err:
        logger.error("Plex server not found: %s", err)
        raise RuntimeError("Plex server could not be reached. Check PLEX_URL.") from err


def fetch_movies(library) -> Iterable:
    try:
        logger.info("Fetching movies from library '%s'", library.title)
        return library.all(libtype="movie")
    except NotFound as err:
        logger.error("Library '%s' not found or empty: %s", library.title, err)
        raise RuntimeError(
            f"Plex library '{library.title}' is unavailable or contains no movies."
        ) from err


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[\s._-]+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def filename_from_path(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def heuristic_match(title: str, filename: str) -> bool:
    norm_title = normalize(title)
    norm_filename = normalize(filename)
    logger.trace("Heuristic compare title='%s' filename='%s'", norm_title, norm_filename)
    if not norm_title or not norm_filename:
        return False

    if norm_title == norm_filename:
        return True

    if norm_title in norm_filename:
        return True

    if norm_filename in norm_title:
        return True

    return False


def make_openai_client() -> OpenAIChatClient:
    model = os.getenv("OPENAI_MODEL", "gpt-5-mini")
    try:
        temperature = float(os.getenv("OPENAI_TEMPERATURE", "1"))
    except ValueError:
        temperature = 1.0
    client = OpenAIChatClient(model=model, temperature=temperature)
    logger.info("OpenAI client ready (model=%s, temperature=%s)", model, temperature)
    return client


def format_movie_identifier(title: str, year: int | None) -> str:
    if year:
        return f"{title} ({year})"
    return title


def build_prompt(filename: str, title: str, year: int | None, extra: Dict[str, str]) -> str:
    details = {
        "filename": filename,
        "title": title,
        "year": year or "",
        "originalTitle": extra.get("originalTitle", ""),
        "summary": extra.get("summary", ""),
        "folderName": extra.get("folderName", ""),
        "fullPath": extra.get("fullPath", ""),
    }
    folder_line = f"Containing Folder: {details['folderName']}\n" if details["folderName"] else ""
    full_path_line = f"Full Path: {details['fullPath']}\n" if details["fullPath"] else ""
    summary_preview = details["summary"][:400]
    return (
        "You are validating whether a media file name matches the intended movie metadata.\n"
        "Compare the filename (and any folder hints) to the movie title (and year if provided). "
        "Consider common naming conventions, abbreviations, translations, "
        "and the possibility that the filename might include resolution or release group tags.\n\n"
        f"Filename: {details['filename']}\n"
        f"Movie Title: {details['title']}\n"
        f"Release Year: {details['year']}\n"
        f"Original Title: {details['originalTitle']}\n"
        f"{folder_line}"
        f"{full_path_line}"
        f"Summary: {summary_preview}\n\n"
        "Respond strictly in JSON with the following shape:\n"
        '{ "match": true|false, "confidence": 0-1, "reason": "brief explanation" }\n'
        "Set match to true only if you are confident the filename corresponds to the title."
    )


def call_openai_check(
    client: OpenAIChatClient, filename: str, title: str, year: int | None, extra: Dict[str, str]
) -> Tuple[bool, float, str]:
    prompt = build_prompt(filename, title, year, extra)
    logger.debug("Querying OpenAI for '%s' vs '%s'", filename, title)
    logger.trace("OpenAI prompt: %s", prompt)

    messages = [
        {"role": "system", "content": "You validate movie filenames for Plex libraries."},
        {"role": "user", "content": prompt},
    ]

    try:
        output_text = client.complete(messages)
    except RuntimeError as exc:
        logger.error("OpenAI API error: %s", exc)
        return False, 0.0, str(exc)

    if not output_text:
        logger.warning("OpenAI returned no text for '%s'", filename)
        return False, 0.0, "Empty response from OpenAI"

    output_text = output_text.strip()
    logger.trace("OpenAI raw response: %s", output_text)
    try:
        data = json.loads(output_text)
    except json.JSONDecodeError:
        logger.warning("Failed to parse OpenAI response for '%s': %s", filename, output_text)
        return False, 0.0, f"Unparseable response: {output_text}"

    match = bool(data.get("match"))
    confidence = float(data.get("confidence", 0.0))
    reason = str(data.get("reason", "")).strip() or "No reason provided."
    logger.debug(
        "OpenAI decision for '%s': match=%s confidence=%.2f reason=%s",
        filename,
        match,
        confidence,
        reason,
    )
    return match, confidence, reason


def collect_movie_paths(movie) -> List[str]:
    paths: List[str] = []
    media_list = getattr(movie, "media", None) or []
    for media in media_list:
        for part in getattr(media, "parts", []) or []:
            path = getattr(part, "file", None)
            if path:
                paths.append(path)
                logger.trace("Found media path for '%s': %s", movie.title, path)
    return paths


def evaluate_movie_files(
    client: OpenAIChatClient,
    movie,
    cache: Dict[str, Dict[str, Any]],
    refresh_cache: bool,
) -> Tuple[List[Dict[str, object]], bool]:
    title = getattr(movie, "title", "Unknown Title")
    year = getattr(movie, "year", None)
    original_title = getattr(movie, "originalTitle", "")
    summary = getattr(movie, "summary", "")

    results: List[Dict[str, object]] = []
    cache_modified = False
    paths = collect_movie_paths(movie)

    for path in paths:
        base = filename_from_path(path)
        cache_entry = cache.get(path)

        if cache_entry and cache_entry.get("match") and not refresh_cache:
            logger.debug(
                "Skipping '%s' (cached match with confidence %.2f).",
                path,
                cache_entry.get("confidence", 0.0),
            )
            results.append(
                {
                    "path": path,
                    "filename": base,
                    "match": True,
                    "confidence": float(cache_entry.get("confidence", 1.0)),
                    "reason": cache_entry.get("reason", "Cached confirmation."),
                    "method": cache_entry.get("method", "cache"),
                    "cached": True,
                }
            )
            continue

        if heuristic_match(title, base):
            logger.debug(
                "Heuristic match succeeded for '%s' -> '%s'", title, base
            )
            result = {
                "path": path,
                "filename": base,
                "match": True,
                "confidence": 1.0,
                "reason": "Heuristic match (title matches filename).",
                "method": "heuristic",
                "cached": False,
            }
            results.append(result)
            cache[path] = {
                "match": True,
                "confidence": result["confidence"],
                "reason": result["reason"],
                "method": result["method"],
                "title": title,
                "year": year,
                "verified_at": datetime.utcnow().isoformat() + "Z",
            }
            cache_modified = True
            continue

        folder_hint = os.path.basename(os.path.dirname(path))
        match, confidence, reason = call_openai_check(
            client,
            base,
            title,
            year,
            {
                "originalTitle": original_title,
                "summary": summary,
                "folderName": folder_hint,
                "fullPath": path,
            },
        )
        logger.debug(
            "OpenAI evaluated '%s' -> '%s': match=%s confidence=%.2f",
            title,
            base,
            match,
            confidence,
        )
        result = {
            "path": path,
            "filename": base,
            "match": match,
            "confidence": confidence,
            "reason": reason,
            "method": "openai",
            "cached": False,
        }
        results.append(result)

        if match:
            cache[path] = {
                "match": match,
                "confidence": confidence,
                "reason": reason,
                "method": "openai",
                "title": title,
                "year": year,
                "verified_at": datetime.utcnow().isoformat() + "Z",
            }
            cache_modified = True
        elif cache_entry:
            cache.pop(path, None)
            cache_modified = True

    return results, cache_modified


def main(argv: List[str] | None = None) -> None:
    try:
        args = parse_args(argv)
        configure_logging(args.log_level)
        logger.info("Starting filename validation with log level %s", args.log_level.upper())

        load_env_file(ENV_FILE)
        env = get_required_env(REQUIRED_ENV_KEYS)

        cache_path = Path(args.cache_file)
        cache = load_cache(cache_path)
        if args.refresh_cache:
            logger.info("Refresh cache requested; existing confirmations will be revalidated.")

        plex = connect_to_plex(env["PLEX_URL"], env["PLEX_TOKEN"])
        try:
            library = plex.library.section(env["PLEX_LIBRARY_NAME"])
        except NotFound as err:
            raise RuntimeError(
                f"Plex library '{env['PLEX_LIBRARY_NAME']}' was not found."
            ) from err

        movies = list(fetch_movies(library))
        logger.info("Discovered %d movie(s) for validation", len(movies))
        client = make_openai_client()
        logger.debug("OpenAI client initialized")

        total_checks = 0
        flagged = []
        cache_dirty = False

        for movie in movies:
            logger.info("Evaluating '%s'", getattr(movie, "title", "Unknown Title"))
            movie_results, cache_changed = evaluate_movie_files(
                client, movie, cache, args.refresh_cache
            )
            total_checks += len(movie_results)
            cache_dirty = cache_dirty or cache_changed

            mismatches = [res for res in movie_results if not res["match"]]
            if mismatches:
                logger.warning(
                    "Potential mismatch found for '%s' (%d file(s))",
                    getattr(movie, "title", "Unknown Title"),
                    len(mismatches),
                )
                flagged.append(
                    {
                        "movie": format_movie_identifier(
                            getattr(movie, "title", "Unknown Title"), getattr(movie, "year", None)
                        ),
                        "mismatches": mismatches,
                    }
                )

        logger.info(
            "Checked %d file(s) across %d movie(s).", total_checks, len(movies)
        )

        confirmed_matches = sum(1 for info in cache.values() if info.get("match"))
        if cache_dirty or args.refresh_cache or not cache_path.exists():
            save_cache(cache_path, cache)
        else:
            logger.debug("Cache unchanged; not writing to disk.")
        logger.info("Cached confirmed matches: %d", confirmed_matches)

        if not flagged:
            logger.info("All files appear to match their metadata.")
            return

        logger.warning("Potential mismatches detected:")
        for entry in flagged:
            logger.warning("- %s", entry["movie"])
            for item in entry["mismatches"]:
                confidence_pct = int(round(item["confidence"] * 100))
                logger.warning("    File: %s", item["path"])
                logger.warning("      Confidence: %s%%", confidence_pct)
                logger.warning("      Reason: %s", item["reason"])

    except Exception as exc:
        sys.stderr.write(f"Error: {exc}\n")
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])

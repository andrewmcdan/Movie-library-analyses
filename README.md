# Movie Library Analyses

A collection of Python tools for analyzing and validating Plex movie libraries.

## Overview

This repository contains two main tools for Plex media server users:

1. **Plex Movie Analytics GUI** (`main.py`) - A comprehensive analytics dashboard for your Plex movie library
2. **Plex Movie Filename Validator** (`names.py`) - An AI-powered tool to validate movie filenames against metadata

## Features

### Plex Movie Analytics GUI (`main.py`)

Provides detailed analytics about your Plex movie library through an easy-to-use GUI:

- **Runtime Analysis**: Average/median duration, longest/shortest movies, epic films statistics
- **Ratings Analytics**: Rating distributions, highest/lowest rated titles, actor/director ratings
- **Release Timeline**: Release year trends, decade analysis, oldest/newest titles
- **Library Growth**: Addition dates, growth trends, recent additions tracking
- **Watch Activity**: View counts, most rewatched titles, watch history
- **Talent & Genres**: Top actors, directors, and genre breakdowns
- **Audience Insights**: Content ratings, audio languages, countries of origin
- **Collections & Sources**: Collection membership, metadata source tracking (IMDb, TMDb, etc.)
- **Technical Details**: Video resolutions, HDR/SDR distribution, audio channel layouts
- **Metadata Quality**: Identifies missing summaries, ratings, posters, and genres

**Additional Features**:
- Multi-threaded analysis for fast processing of large libraries
- Real-time progress updates
- Tabbed interface for easy navigation

### Plex Movie Filename Validator (`names.py`)

Validates that your movie filenames correctly match their Plex metadata:

- **Intelligent Matching**: Uses heuristic analysis and OpenAI API for accurate validation
- **Caching System**: Stores confirmed matches to avoid repeated API calls
- **Flexible Validation**: Supports various filename formats and naming conventions
- **Detailed Reporting**: Provides confidence scores and explanations for each match
- **Folder Context**: Considers parent folder names for better accuracy

## Requirements

### System Requirements
- Python 3.10 or higher
- Windows, macOS, or Linux
- Tkinter (usually included with Python)

### Python Dependencies

**For Plex Movie Analytics GUI (`main.py`)**:
```
plexapi
tkinter (built-in with most Python installations)
```

**For Plex Movie Filename Validator (`names.py`)**:
```
plexapi
openai
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/andrewmcdan/Movie-library-analyses.git
   cd Movie-library-analyses
   ```

2. **Install dependencies**:
   ```bash
   pip install plexapi openai
   ```

   Or create a virtual environment (recommended):
   ```bash
   python -m venv .venv
   
   # On Windows:
   .venv\Scripts\activate
   
   # On macOS/Linux:
   source .venv/bin/activate
   
   pip install plexapi openai
   ```

## Configuration

Both scripts require a `.env` file in the same directory with your Plex credentials.

Create a `.env` file with the following contents:

```env
PLEX_URL=http://your-plex-server:32400
PLEX_TOKEN=your_plex_token_here
PLEX_LIBRARY_NAME=Movies

# Required only for names.py:
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=1
```

### Finding Your Plex Token

To find your Plex authentication token:

1. Open a movie in your Plex web interface
2. Click the "..." menu and select "Get Info"
3. Click "View XML"
4. Look for `X-Plex-Token` in the URL

Alternatively, see the [official Plex guide](https://support.plex.tv/articles/204059436-finding-an-authentication-token-x-plex-token/).

### OpenAI API Key (for names.py only)

If you plan to use the filename validator:

1. Sign up at [OpenAI](https://platform.openai.com/)
2. Generate an API key from your account settings
3. Add it to your `.env` file

## Usage

### Plex Movie Analytics GUI

Run the analytics tool:

```bash
python main.py
```

**Options**:
```bash
python main.py --log-level debug    # Enable debug logging
python main.py --workers 8          # Use 8 worker threads (default: half of CPU cores)
```

The GUI will open and automatically begin analyzing your library. Results are displayed in multiple tabs:
- Runtime Profile
- Ratings
- Release Timeline
- Library Growth
- Watch Activity
- Talent & Genres
- Audience & Origins
- Collections & Sources
- Technical Snapshot
- Metadata Gaps

### Plex Movie Filename Validator

Run the filename validator:

```bash
python names.py
```

**Options**:
```bash
python names.py --log-level debug                    # Enable debug logging
python names.py --cache-file my_matches.json         # Use custom cache file
python names.py --refresh-cache                      # Revalidate all cached entries
python names.py --force-confirm                      # Override mismatches and confirm all
```

The tool will:
1. Connect to your Plex server
2. Check each movie file against its metadata
3. Report any potential mismatches
4. Save confirmed matches to cache for future runs

## Logging Levels

Both scripts support multiple logging levels:
- `trace`: Most verbose, includes all debug information
- `debug`: Detailed debugging information
- `info`: General informational messages (default)
- `warn`: Warning messages only
- `error`: Error messages only

Example:
```bash
python main.py --log-level trace
```

## Troubleshooting

### Tkinter Not Available
If you see "Tkinter is not available", install it:

**Ubuntu/Debian**:
```bash
sudo apt-get install python3-tk
```

**macOS** (using Homebrew):
```bash
brew install python-tk
```

### Plex Connection Issues
- Verify your `PLEX_URL` includes the correct protocol (`http://` or `https://`)
- Ensure your Plex server is accessible from your machine
- Check that your `PLEX_TOKEN` is valid
- Confirm the `PLEX_LIBRARY_NAME` matches exactly (case-sensitive)

### OpenAI API Issues
- Verify your API key is valid
- Check your OpenAI account has available credits
- The default model `gpt-4o-mini` is cost-effective; you can use other models like `gpt-4` by changing `OPENAI_MODEL` in `.env`

## Performance Tips

### For Large Libraries (main.py)
- The analytics tool uses parallel processing by default
- Increase workers with `--workers` flag for faster processing on multi-core systems
- Processing time scales with library size (expect ~1-5 minutes for 1000+ movies)

### For Filename Validation (names.py)
- Use the cache system to avoid repeated API calls (saves time and money)
- Initial run will be slower due to OpenAI API calls
- Subsequent runs with cache are nearly instant for previously validated files
- Consider using `--force-confirm` if you trust your existing filenames

## License

This project is provided as-is for personal use. Please ensure you comply with Plex's and OpenAI's terms of service when using these tools.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Author

Created by andrewmcdan

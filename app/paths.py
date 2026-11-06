"""Where the server keeps its data: records, cache, settings, services, glossaries, models and layout results.

The directory is data/ in the project by default. The environment variable DOC_TRANSLATOR_DATA moves it elsewhere;
the tests point it at a temporary directory, so importing the app never reads or rewrites the records of a server
running from the same checkout.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("DOC_TRANSLATOR_DATA") or ROOT / "data").expanduser()

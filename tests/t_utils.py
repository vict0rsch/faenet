import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT))
os.environ["SQLALCHEMY_SILENCE_UBER_WARNING"] = "1"

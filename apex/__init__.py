import sys
from pathlib import Path

# Ensure root workspace path is in sys.path so 'plugins' package is importable
_root_dir = str(Path(__file__).resolve().parent.parent)
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

__version__ = "0.1.0"
__license__ = "Apache-2.0"
__author__ = "APEX-Track Contributors"


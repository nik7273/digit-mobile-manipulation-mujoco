"""Repository resources, independent of the current working directory."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = ROOT / "assets" / "package_scene.xml"
HOLD_SCENE = ROOT / "assets" / "standing_hold.xml"

"""Prepare project directories without replacing any existing files."""

from pathlib import Path

FOLDERS = ("data", "notebooks", "scripts", "models", "dashboard")


def prepare_directories(base: Path) -> None:
    for folder in FOLDERS:
        (base / folder).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    prepare_directories(Path(__file__).resolve().parent)
    print("Project directories ready. Existing files were preserved.")
    print("See docs/INSTALLATION.md to create an isolated demo environment.")

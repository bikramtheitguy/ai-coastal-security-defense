import json
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from setup_project import FOLDERS, prepare_directories

ROOT = Path(__file__).resolve().parents[1]


def test_setup_preserves_existing_files_and_is_repeatable(tmp_path):
    for name in ("README.md", "requirements.txt"):
        (tmp_path / name).write_text("keep this exact content", encoding="utf-8")
    prepare_directories(tmp_path)
    prepare_directories(tmp_path)
    for folder in FOLDERS:
        assert (tmp_path / folder).is_dir()
    for name in ("README.md", "requirements.txt"):
        assert (tmp_path / name).read_text(encoding="utf-8") == "keep this exact content"


def test_setup_cli_uses_script_directory(tmp_path):
    project = tmp_path / "project with spaces"
    project.mkdir()
    shutil.copy2(ROOT / "setup_project.py", project)
    (project / "README.md").write_text("preserve", encoding="utf-8")
    subprocess.run([sys.executable, str(project / "setup_project.py")], cwd=tmp_path, check=True)
    assert (project / "dashboard").is_dir()
    assert not (tmp_path / "dashboard").exists()
    assert (project / "README.md").read_text(encoding="utf-8") == "preserve"


@pytest.mark.skipif(sys.platform == "win32", reason="Bash wrapper is exercised on Linux CI")
def test_shell_wrapper_handles_spaces_and_preserves_files(tmp_path):
    project = tmp_path / "project with spaces"
    project.mkdir()
    for name in ("setup.sh", "setup_project.py"):
        shutil.copy2(ROOT / name, project)
    (project / "requirements.txt").write_text("preserve", encoding="utf-8")
    subprocess.run(["bash", str(project / "setup.sh")], cwd=tmp_path, check=True)
    assert (project / "dashboard").is_dir()
    assert (project / "requirements.txt").read_text(encoding="utf-8") == "preserve"


def test_security_configuration_is_enabled():
    config = tomllib.loads((ROOT / ".streamlit/config.toml").read_text(encoding="utf-8"))
    assert config["server"]["enableCORS"] is True
    assert config["server"]["enableXsrfProtection"] is True
    container = json.loads((ROOT / ".devcontainer/devcontainer.json").read_text(encoding="utf-8"))
    command = container["postAttachCommand"]["server"]
    assert "--server.enableCORS=true" in command
    assert "--server.enableXsrfProtection=true" in command
    assert "pip3 install --user" not in json.dumps(container)

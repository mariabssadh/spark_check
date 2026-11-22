from pathlib import Path

ROOT = Path(__file__).parent.parent


def test_gitignore_blocks_env_file():
    gitignore = (ROOT / ".gitignore").read_text()
    assert ".env" in gitignore.splitlines()


def test_dev_requirements_list_ci_tools():
    requirements = (ROOT / "requirements-dev.txt").read_text().splitlines()
    assert "pytest" in requirements
    assert "ruff" in requirements
    
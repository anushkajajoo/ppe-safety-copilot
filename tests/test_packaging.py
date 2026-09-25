"""
Tests for the things that make the project runnable by someone else: the container, the CI
pipeline, the dependency list and the documentation set.

These are cheap tests for expensive mistakes. A Dockerfile that runs as root, a requirements
file that grew a dependency nothing imports, or a missing document are all things a reviewer
notices immediately and a test can notice first.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(*parts) -> str:
    return (ROOT.joinpath(*parts)).read_text(encoding="utf-8")


# ------------------------------------------------------------------- container
def test_the_container_recipe_exists():
    assert (ROOT / "Dockerfile").exists()
    assert (ROOT / ".dockerignore").exists()


def test_the_container_does_not_run_as_root():
    """A process watching a camera has no business being root."""
    dockerfile = read("Dockerfile")
    assert re.search(r"^USER\s+app", dockerfile, re.M)
    assert "useradd" in dockerfile


def test_the_container_is_healthy_only_when_the_api_answers():
    dockerfile = read("Dockerfile")
    assert "HEALTHCHECK" in dockerfile and "/health" in dockerfile


def test_the_container_uses_headless_opencv():
    """There is no display inside a container; the GUI build would pull in X11 for nothing."""
    assert "opencv-python-headless" in read("Dockerfile")


def test_secrets_and_data_never_enter_the_image():
    ignored = read(".dockerignore")
    for entry in (".env", "data", "datasets", ".git"):
        assert entry in ignored


# -------------------------------------------------------------------------- CI
def test_the_pipeline_exists_and_covers_the_four_checks():
    ci = read(".github", "workflows", "ci.yml")
    for job in ("tests:", "lint:", "security:", "container:"):
        assert job in ci, f"CI is missing the {job.strip(':')} job"


def test_the_pipeline_scans_dependencies_and_refuses_committed_secrets():
    ci = read(".github", "workflows", "ci.yml")
    assert "pip-audit" in ci
    assert "-f .env" in ci      # the build fails if a .env was ever committed


def test_the_pipeline_runs_the_real_test_suite():
    assert "python -m pytest" in read(".github", "workflows", "ci.yml")


# ---------------------------------------------------------------- dependencies
def test_no_dependency_is_listed_that_nothing_imports():
    """
    streamlit, plotly and pandas were dropped when the dashboard became plain HTML (D-023).
    An unused dependency is still an install, a licence and something the scanner checks.
    """
    requirements = read("requirements.txt")
    active = [line.split("#")[0].strip().lower()
              for line in requirements.splitlines()
              if line.strip() and not line.strip().startswith("#")]
    for dropped in ("streamlit", "plotly", "pandas"):
        assert not any(entry.startswith(dropped) for entry in active), \
            f"{dropped} is back in requirements.txt but nothing imports it"


def test_the_upload_dependency_is_pinned():
    """python-multipart is not imported by name, so it is easy to lose - and /predict dies."""
    assert "python-multipart" in read("requirements.txt")


def test_the_env_file_can_never_be_committed():
    """
    A local .env is correct and expected - it holds this machine's settings. What must never
    happen is it reaching the repository, so the guard is .gitignore (and the CI job that
    fails the build if a checkout contains one), not the absence of the file on disk.
    """
    ignored = [line.strip() for line in read(".gitignore").splitlines()]
    assert ".env" in ignored, ".env must be gitignored"
    assert "-f .env" in read(".github", "workflows", "ci.yml")


# ------------------------------------------------------------------------ docs
def test_every_mandatory_document_exists():
    for name in ("model_card.md", "threat_model.md", "problem_brief.md",
                 "test_strategy.md", "data_dictionary.md", "decisions.md",
                 "viva_qa.md", "presentation.md", "admin_guide.md"):
        assert (ROOT / "docs" / name).exists(), f"docs/{name} is missing"
    assert (ROOT / "docs" / "architecture" / "architecture.md").exists()
    assert (ROOT / "docs" / "evaluation" / "system_evaluation.md").exists()


def test_release_notes_record_the_current_milestone():
    changelog = read("CHANGELOG.md")
    assert "## [0.5.0]" in changelog
    assert "Known open" in changelog, "release notes must state what is still open"


# ------------------------------------------------------------------ frontend
def element_ids(html: str):
    """Static ids in the markup - not the ones JavaScript builds into table rows."""
    import re
    without_templates = re.sub(r"\$\{[^}]*\}", "TEMPLATE", html)
    return re.findall(r'(?<!data-)id="([^"]+)"', without_templates)


def test_no_page_has_two_elements_with_the_same_id():
    """
    The bug this catches: the health panel used id="health", which the top bar already
    used for its status pill. getElementById found the pill first and rendered twelve
    system-health tiles into the header. A duplicate id is not a style question - it
    silently sends content to the wrong element.
    """
    from collections import Counter
    for page in ("dashboard.html", "live.html"):
        ids = [i for i in element_ids(read("frontend", page)) if "TEMPLATE" not in i]
        duplicates = {name: count for name, count in Counter(ids).items() if count > 1}
        assert not duplicates, f"{page} reuses these ids: {duplicates}"


def test_every_id_the_dashboard_script_writes_to_exists_in_the_markup():
    """A typo in $('...') fails silently in a browser; here it fails loudly."""
    import re
    html = read("frontend", "dashboard.html")
    declared = set(element_ids(html))
    used = set(re.findall(r"\$\('([A-Za-z0-9_]+)'\)", html))
    missing = sorted(used - declared)
    assert not missing, f"the script writes to ids that do not exist: {missing}"

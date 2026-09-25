"""Tests for scripts/make_colab_bundle.py and the Colab notebook."""
import json
from pathlib import Path

from scripts import make_colab_bundle as bundler

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "training" / "colab_train.ipynb"


def test_bundle_includes_code_and_dataset_only():
    assert bundler.INCLUDE == ["training", "scripts", "datasets/ppe4"]
    for junk in ["__pycache__", "_incoming"]:
        assert junk in bundler.SKIP_PARTS      # never ship caches or transfer zips


def test_notebook_is_valid_json_with_cells():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    assert len(nb["cells"]) > 10
    for cell in nb["cells"]:
        assert cell["cell_type"] in {"markdown", "code"}
        assert isinstance(cell["source"], list)


def _code(nb):
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


def test_notebook_asks_for_a_gpu_and_uses_our_settings():
    code = _code(json.loads(NOTEBOOK.read_text(encoding="utf-8")))
    assert "torch.cuda.is_available()" in code          # fails fast without a GPU
    assert "--epochs 30" in code
    assert "--mosaic 0.0" in code                       # our images are already mosaics
    assert "--split val" in code and "--split test" in code
    assert "--resume" in code                           # survives a Colab disconnect


def test_notebook_shell_commands_are_single_line():
    """IPython's ! magic is line-based; a backslash continuation would break it."""
    code = _code(json.loads(NOTEBOOK.read_text(encoding="utf-8")))
    for line in code.splitlines():
        if line.strip().startswith("!"):
            assert not line.rstrip().endswith("\\"), line

"""
Tests for the training command line (training/train.py).

These check the SETTINGS, not the training itself - training needs a GPU and
minutes of time, so it is not run in the test suite. `train.py` imports torch
and ultralytics inside main(), so importing the module here stays fast.
"""
from training.train import build_parser


def test_defaults_are_the_lightweight_ones():
    args = build_parser().parse_args([])
    assert args.epochs == 30                 # brief: 20-40 epochs, not 100+
    assert args.batch == 8                   # fits 4 GB of VRAM
    assert args.imgsz == 640
    assert args.patience == 8                # early stopping
    assert args.workers <= 4                 # Windows dataloader
    assert args.device == "0"


def test_mosaic_is_off_by_default():
    """Our training images are already Roboflow mosaics - see docs/decisions.md D-010a."""
    assert build_parser().parse_args([]).mosaic == 0.0


def test_mosaic_can_be_turned_back_on():
    assert build_parser().parse_args(["--mosaic", "1.0"]).mosaic == 1.0


def test_data_points_at_our_dataset():
    args = build_parser().parse_args([])
    assert args.data.replace("\\", "/").endswith("datasets/ppe4/data.yaml")


def test_resume_is_a_flag():
    assert build_parser().parse_args([]).resume is False
    assert build_parser().parse_args(["--resume"]).resume is True


def test_small_smoke_run_can_be_requested():
    args = build_parser().parse_args(["--epochs", "3", "--fraction", "0.1", "--name", "smoke"])
    assert (args.epochs, args.fraction, args.name) == (3, 0.1, "smoke")

"""
Tests for run.py - the single start command.

run.py imports uvicorn inside main(), so importing the module here is instant and
needs no server.
"""
import socket

import run


def test_defaults_are_local_only():
    args = run.build_parser().parse_args([])
    assert args.host == "127.0.0.1"          # not 0.0.0.0: the demo stays on this machine
    assert args.port == 8000
    assert args.no_browser is False and args.reload is False


def test_options_can_be_changed():
    args = run.build_parser().parse_args(["--port", "8001", "--no-browser", "--reload"])
    assert (args.port, args.no_browser, args.reload) == (8001, True, True)


def test_urls_point_at_the_pages_we_actually_serve():
    links = run.urls("127.0.0.1", 8000)
    assert links["dashboard"] == "http://127.0.0.1:8000/dashboard"
    assert links["live"] == "http://127.0.0.1:8000/live"
    assert links["docs"].endswith("/docs") and links["health"].endswith("/health")


def test_port_is_free_detects_a_busy_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        busy_port = server.getsockname()[1]
        assert run.port_is_free("127.0.0.1", busy_port) is False
    assert run.port_is_free("127.0.0.1", busy_port) is True


def test_banner_lists_the_pages_and_the_privacy_line():
    text = run.banner("127.0.0.1", 8000, weights_found=True)
    assert "/dashboard" in text and "/live" in text
    assert "No frames are uploaded anywhere." in text
    assert "WARNING: model not found" not in text


def test_banner_warns_when_the_model_is_missing():
    assert "WARNING: model not found" in run.banner("127.0.0.1", 8000, weights_found=False)

"""
Two status constants, in one place.

Starlette renamed `HTTP_422_UNPROCESSABLE_ENTITY` to `HTTP_422_UNPROCESSABLE_CONTENT` and
`HTTP_413_REQUEST_ENTITY_TOO_LARGE` to `HTTP_413_CONTENT_TOO_LARGE`. The old names still
work but emit a DeprecationWarning on every run; the new ones do not exist on older
versions. Since this project has to install from requirements.txt on a machine that is not
mine, it uses whichever the installed version provides - and keeps the name rather than
scattering bare numbers through the routes.
"""
import warnings

from fastapi import status


def _pick(new_name: str, old_name: str, code: int) -> int:
    """
    Reading the old name is itself what triggers Starlette's DeprecationWarning, so the
    lookup is done with warnings suppressed: this module exists precisely to stop using
    the deprecated name, and it should not print a warning while doing so. Only this one
    lookup is silenced - anything else the project deprecates still shows up.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return getattr(status, new_name, getattr(status, old_name, code))


UNPROCESSABLE = _pick("HTTP_422_UNPROCESSABLE_CONTENT", "HTTP_422_UNPROCESSABLE_ENTITY", 422)
TOO_LARGE = _pick("HTTP_413_CONTENT_TOO_LARGE", "HTTP_413_REQUEST_ENTITY_TOO_LARGE", 413)

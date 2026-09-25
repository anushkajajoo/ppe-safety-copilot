"""
Sign-in, sign-out, and "who am I".

    POST /api/v1/auth/login    {username, password} -> sets an HttpOnly session cookie
    POST /api/v1/auth/logout   clears it
    GET  /api/v1/auth/me       the current user and role, or signed_in: false

The dependency `require_role(...)` in this module is what every protected route uses. It is
deliberately the only way in: a route either declares the role it needs or is open, and
which one it is can be read off the route definition.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field

from server.auth import (ADMIN, AuthError, COOKIE_NAME, SESSION_HOURS, SUPERVISOR, VIEWER,
                         authenticate, make_token, read_token, role_allows)

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


class Credentials(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=1, max_length=256)


def current_user(request: Request) -> Optional[dict]:
    """The signed-in user, or None. Never raises - callers decide whether that matters."""
    token = request.cookies.get(COOKIE_NAME, "")
    if not token:
        return None
    try:
        return read_token(token, request.app.state.session_secret)
    except AuthError:
        return None


def require_role(required: str):
    """
    Dependency factory: `Depends(require_role(SUPERVISOR))`.

    401 when nobody is signed in (you need to authenticate), 403 when someone is but their
    role is too low (authentication worked, authorisation did not). Keeping those apart is
    what lets the interface say the right thing.
    """
    def dependency(request: Request) -> dict:
        user = current_user(request)
        if user is None:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED,
                                detail="sign in to do this")
        if not role_allows(user.get("r", ""), required):
            raise HTTPException(status.HTTP_403_FORBIDDEN,
                                detail=f"this needs the {required} role; you are {user.get('r')}")
        return user
    return dependency


@router.post("/login")
def login(credentials: Credentials, response: Response, request: Request):
    try:
        user = authenticate(request.app.state.users, credentials.username, credentials.password)
    except AuthError as exc:
        # One message for both "no such user" and "wrong password", on purpose.
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail=str(exc))

    token = make_token(user, request.app.state.session_secret)
    response.set_cookie(COOKIE_NAME, token, httponly=True, samesite="strict",
                        max_age=int(SESSION_HOURS * 3600), path="/")
    return {"signed_in": True, "username": user.username, "role": user.role,
            "display_name": user.label()}


@router.post("/logout")
def logout(response: Response):
    response.delete_cookie(COOKIE_NAME, path="/")
    return {"signed_in": False}


@router.get("/me")
def me(request: Request):
    user = current_user(request)
    if user is None:
        return {"signed_in": False, "role": None,
                "roles": {"viewer": "see the site", "supervisor": "decide on proposals",
                          "admin": "configuration and maintenance"}}
    return {"signed_in": True, "username": user["u"], "role": user["r"],
            "display_name": user.get("n", user["u"]),
            "can_decide": role_allows(user["r"], SUPERVISOR),
            "can_administer": role_allows(user["r"], ADMIN)}

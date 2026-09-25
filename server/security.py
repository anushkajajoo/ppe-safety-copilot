"""
Edge-device authentication (MVP): the edge sends a shared secret in the
`X-API-Key` header. Reviewer/admin roles for the dashboard come on Day 5.

hmac.compare_digest is used instead of == so the comparison takes the same time
whether the first character or the last one is wrong (prevents timing attacks).
"""
import hmac

from fastapi import Header, HTTPException, Request, status


def require_edge_key(request: Request, x_api_key: str = Header(default="", alias="X-API-Key")) -> str:
    expected = request.app.state.settings.edge_api_key
    if not x_api_key or not hmac.compare_digest(x_api_key.encode(), expected.encode()):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid or missing X-API-Key")
    return "edge"

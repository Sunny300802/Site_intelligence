"""
web/auth.py
===========
Password hashing (stdlib pbkdf2 - no extra dependency) and the session
check used by protected routes.
"""
import os
import hmac
import hashlib

ITERATIONS = 200_000


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, ITERATIONS)
    return f"pbkdf2_sha256${ITERATIONS}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _algo, iters, salt_hex, hash_hex = stored.split("$")
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(),
                                 bytes.fromhex(salt_hex), int(iters))
        return hmac.compare_digest(dk.hex(), hash_hex)
    except Exception:
        return False


def current_user(request):
    """The signed-in username, or None. `request` is a Starlette Request;
    it is not type-annotated so this module stays importable by the
    command-line tools without pulling in the whole web stack."""
    return request.session.get("user")

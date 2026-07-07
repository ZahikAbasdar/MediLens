"""
auth.py
=======
Handles everything related to WHO the user is:
  - Hashing and verifying passwords (never store plain text passwords)
  - Creating JWT access tokens on login
  - Decoding/validating JWT tokens on every protected request
  - A reusable FastAPI dependency `get_current_user` that any route
    can use to require login

HOW JWT LOGIN WORKS (for the beginner reading this):
1. User sends email+password to /auth/login.
2. We verify the password against the stored hash.
3. We create a JWT ("JSON Web Token") - a signed string that encodes
   "this is user #5, and this token is valid until <time>".
4. The client stores that token and sends it in the
   `Authorization: Bearer <token>` header on every future request.
5. `get_current_user` decodes the token, checks the signature and
   expiry, and looks up the matching user in the database.

Because the token is cryptographically SIGNED with SECRET_KEY, a
user cannot forge or tamper with it without knowing that key.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from config import settings
from database import get_db
from models import User

logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# Password hashing
# ------------------------------------------------------------
# bcrypt automatically handles salting, so two users with the same
# password get completely different hashes stored in the database.
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(plain_password: str) -> str:
    """Turns a plain-text password into a secure one-way hash."""
    return pwd_context.hash(plain_password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Checks a plain-text password against a stored hash. Returns True/False."""
    return pwd_context.verify(plain_password, hashed_password)


# ------------------------------------------------------------
# JWT token creation
# ------------------------------------------------------------
def create_access_token(data: dict, expires_minutes: Optional[int] = None) -> str:
    """
    Builds a signed JWT string.

    `data` typically looks like {"sub": "user@example.com"} - "sub"
    (subject) is the standard JWT field for "who this token is about".

    We add an "exp" (expiry) claim so tokens automatically become
    invalid after ACCESS_TOKEN_EXPIRE_MINUTES, even if never logged out.
    """
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=expires_minutes or settings.ACCESS_TOKEN_EXPIRE_MINUTES
    )
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> dict:
    """
    Decodes and validates a JWT. Raises JWTError if the signature is
    invalid or the token has expired. jose handles both checks for us.
    """
    return jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])


# ------------------------------------------------------------
# FastAPI dependency: get_current_user
# ------------------------------------------------------------
# `OAuth2PasswordBearer` tells FastAPI's auto-generated docs (Swagger
# UI at /docs) that this API uses Bearer token auth, and it knows how
# to extract the token string from the Authorization header for us.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    """
    The core auth dependency. Any protected route declares:

        def my_route(current_user: User = Depends(get_current_user)):

    FastAPI will:
    1. Extract the bearer token from the request header.
    2. Run this function, which decodes the token and fetches the
       matching User row from the database.
    3. If anything fails, raise a 401 Unauthorized automatically -
       the route function body never even runs.
    4. If it succeeds, hand the real `User` object to the route.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = decode_access_token(token)
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise credentials_exception
    return user


def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
    """
    Looks up a user by email and verifies their password.
    Returns the User object on success, or None on failure.
    Used by the /auth/login route.
    """
    user = db.query(User).filter(User.email == email).first()
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


# ------------------------------------------------------------
# Forgot-password: simple in-memory reset code store
# ------------------------------------------------------------
# For a portfolio/beginner project we avoid requiring a real email
# server (SMTP setup is a separate infrastructure concern). Instead
# we generate a reset code and return it directly in the API response
# in DEBUG mode, simulating what a "check your email" flow would do.
# In a real production deployment, replace `_reset_codes` with a
# proper email-sending service and a database-backed, expiring code.
_reset_codes: dict[str, str] = {}


def generate_reset_code(email: str) -> str:
    import secrets
    code = secrets.token_hex(3)  # short 6-character code
    _reset_codes[email] = code
    logger.info("Password reset code generated for %s", email)
    return code


def verify_reset_code(email: str, code: str) -> bool:
    return _reset_codes.get(email) == code


def clear_reset_code(email: str) -> None:
    _reset_codes.pop(email, None)

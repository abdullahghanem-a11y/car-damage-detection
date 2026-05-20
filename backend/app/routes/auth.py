"""
routes_auth.py
--------------
Auth endpoints:
  POST /api/auth/register   — create account
  POST /api/auth/login      — get tokens
  POST /api/auth/refresh    — use httpOnly cookie to get new access token
  POST /api/auth/logout     — clear refresh cookie
  GET  /api/auth/me         — get current user info
  POST /api/auth/feedback   — submit feedback (authenticated users)
"""

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy.orm import Session as DBSession
from datetime import datetime, timezone
from dotenv import load_dotenv
import os

from app.database import get_db
from app.models.user import User, UserRole
from app.models.feedback import Feedback
from app.schemas.auth import (
    RegisterRequest, LoginRequest,
    AuthResponse, TokenResponse, UserResponse,
    FeedbackRequest, FeedbackResponse,
)
from app.services.auth_service import (
    hash_password, verify_password,
    create_access_token, create_refresh_token,
    decode_refresh_token,
    set_refresh_cookie, clear_refresh_cookie,
    get_current_active_user, require_admin,
    REFRESH_COOKIE_NAME, REFRESH_TOKEN_EXPIRE,
)

load_dotenv()

ADMIN_EMAIL    = os.getenv("ADMIN_EMAIL",    "admin@autops.com")
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "change_me_in_production")

router = APIRouter(prefix="/api/auth", tags=["Auth"])


# ── Seed admin on first use ────────────────────────────────────────────────
def ensure_admin_exists(db: DBSession):
    """Create the hardcoded admin account if it doesn't exist yet."""
    existing = db.query(User).filter(User.role == UserRole.admin).first()
    if not existing:
        admin = User(
            email           = ADMIN_EMAIL,
            username        = ADMIN_USERNAME,
            hashed_password = hash_password(ADMIN_PASSWORD),
            role            = UserRole.admin,
            is_active       = True,
            is_verified     = True,
        )
        db.add(admin)
        db.commit()


# ── POST /api/auth/register ────────────────────────────────────────────────
@router.post("/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
def register(body: RegisterRequest, response: Response, db: DBSession = Depends(get_db)):
    # Check duplicates
    if db.query(User).filter(User.email == body.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    if db.query(User).filter(User.username == body.username).first():
        raise HTTPException(status_code=400, detail="Username already taken")

    user = User(
        email           = body.email,
        username        = body.username,
        hashed_password = hash_password(body.password),
        role            = UserRole.user,
        is_active       = True,
        is_verified     = False,   # email verification can be added later
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    access_token  = create_access_token(user.id, user.role)
    refresh_token = create_refresh_token(user.id)
    set_refresh_cookie(response, refresh_token)

    return AuthResponse(access_token=access_token, user=UserResponse.model_validate(user))


# ── POST /api/auth/login ───────────────────────────────────────────────────
@router.post("/login", response_model=AuthResponse)
def login(body: LoginRequest, response: Response, db: DBSession = Depends(get_db)):
    ensure_admin_exists(db)

    # Find by email or username
    user = db.query(User).filter(
        (User.email == body.username_or_email) |
        (User.username == body.username_or_email.lower())
    ).first()

    if not user or not user.hashed_password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not verify_password(body.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is disabled")

    # Update last login
    user.last_login = datetime.now(timezone.utc)
    db.commit()
    db.refresh(user)

    access_token  = create_access_token(user.id, user.role)
    refresh_token = create_refresh_token(user.id)
    set_refresh_cookie(response, refresh_token)

    return AuthResponse(access_token=access_token, user=UserResponse.model_validate(user))


# ── POST /api/auth/refresh ─────────────────────────────────────────────────
@router.post("/refresh", response_model=TokenResponse)
def refresh_token(request: Request, response: Response, db: DBSession = Depends(get_db)):
    """Uses the httpOnly refresh cookie to issue a new access token."""
    token = request.cookies.get(REFRESH_COOKIE_NAME)
    if not token:
        raise HTTPException(status_code=401, detail="No refresh token")

    payload = decode_refresh_token(token)
    user_id = int(payload["sub"])

    user = db.query(User).filter(User.id == user_id).first()
    if not user or not user.is_active:
        clear_refresh_cookie(response)
        raise HTTPException(status_code=401, detail="User not found or disabled")

    # Rotate refresh token
    new_refresh = create_refresh_token(user.id)
    set_refresh_cookie(response, new_refresh)

    return TokenResponse(access_token=create_access_token(user.id, user.role))


# ── POST /api/auth/logout ──────────────────────────────────────────────────
@router.post("/logout")
def logout(response: Response):
    clear_refresh_cookie(response)
    return {"message": "Logged out successfully"}


# ── GET /api/auth/me ───────────────────────────────────────────────────────
@router.get("/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_active_user)):
    return current_user


# ── POST /api/auth/feedback ────────────────────────────────────────────────
@router.post("/feedback", response_model=FeedbackResponse, status_code=status.HTTP_201_CREATED)
def submit_feedback(
    body:         FeedbackRequest,
    db:           DBSession = Depends(get_db),
    current_user: User      = Depends(get_current_active_user),
):
    feedback = Feedback(
        user_id = current_user.id,
        message = body.message,
        rating  = body.rating,
    )
    db.add(feedback)
    db.commit()
    db.refresh(feedback)
    return feedback


# ── GET /api/auth/feedback (admin only) ───────────────────────────────────
@router.get("/feedback", response_model=list[FeedbackResponse])
def get_all_feedback(
    skip:  int     = 0,
    limit: int     = 50,
    db:    DBSession = Depends(get_db),
    admin: User    = Depends(require_admin),
):
    return db.query(Feedback)\
             .order_by(Feedback.created_at.desc())\
             .offset(skip).limit(limit).all()


# ── Google OAuth ───────────────────────────────────────────────────────────
import httpx
import secrets
from urllib.parse import urlencode
from fastapi.responses import RedirectResponse

GOOGLE_CLIENT_ID     = os.getenv("GOOGLE_CLIENT_ID",     "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI  = os.getenv("GOOGLE_REDIRECT_URI",  "http://localhost:8000/api/auth/google/callback")
FRONTEND_URL         = os.getenv("FRONTEND_URL",         "http://localhost:5173")

GOOGLE_AUTH_URL  = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USER_URL  = "https://www.googleapis.com/oauth2/v3/userinfo"

# Simple in-memory state store (prevents CSRF)
_oauth_states: set = set()


# ── GET /api/auth/google ───────────────────────────────────────────────────
@router.get("/google")
def google_login():
    """Redirect user to Google consent screen."""
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=500, detail="Google OAuth not configured")

    state = secrets.token_urlsafe(32)
    _oauth_states.add(state)

    params = urlencode({
        "client_id":     GOOGLE_CLIENT_ID,
        "redirect_uri":  GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope":         "openid email profile",
        "state":         state,
        "access_type":   "offline",
        "prompt":        "select_account",
    })

    return RedirectResponse(f"{GOOGLE_AUTH_URL}?{params}")


# ── GET /api/auth/google/callback ──────────────────────────────────────────
@router.get("/google/callback")
async def google_callback(
    code:  str,
    state: str,
    response: Response,
    db:    DBSession = Depends(get_db),
):
    """Handle Google callback — exchange code for user info, create/find user."""

    # Validate state (CSRF protection)
    if state not in _oauth_states:
        return RedirectResponse(f"{FRONTEND_URL}/login?error=invalid_state")
    _oauth_states.discard(state)

    # Exchange code for tokens
    async with httpx.AsyncClient() as client:
        token_resp = await client.post(GOOGLE_TOKEN_URL, data={
            "code":          code,
            "client_id":     GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "redirect_uri":  GOOGLE_REDIRECT_URI,
            "grant_type":    "authorization_code",
        })

    if token_resp.status_code != 200:
        return RedirectResponse(f"{FRONTEND_URL}/login?error=token_exchange_failed")

    token_data   = token_resp.json()
    access_token = token_data.get("access_token")

    # Get user info from Google
    async with httpx.AsyncClient() as client:
        user_resp = await client.get(
            GOOGLE_USER_URL,
            headers={"Authorization": f"Bearer {access_token}"},
        )

    if user_resp.status_code != 200:
        return RedirectResponse(f"{FRONTEND_URL}/login?error=user_info_failed")

    google_user = user_resp.json()
    google_id   = google_user.get("sub")
    email       = google_user.get("email")
    name        = google_user.get("name", "")
    avatar      = google_user.get("picture")

    if not google_id or not email:
        return RedirectResponse(f"{FRONTEND_URL}/login?error=missing_user_info")

    # Find or create user
    user = db.query(User).filter(User.google_id == google_id).first()

    if not user:
        # Check if email already exists (link accounts)
        user = db.query(User).filter(User.email == email).first()
        if user:
            # Link Google to existing account
            user.google_id  = google_id
            user.avatar_url = avatar
        else:
            # Create new user — generate unique username from name/email
            base_username = name.lower().replace(" ", "_")[:20] or email.split("@")[0][:20]
            username      = base_username
            counter       = 1
            while db.query(User).filter(User.username == username).first():
                username = f"{base_username}_{counter}"
                counter += 1

            user = User(
                email           = email,
                username        = username,
                hashed_password = None,       # no password for Google users
                google_id       = google_id,
                avatar_url      = avatar,
                role            = UserRole.user,
                is_active       = True,
                is_verified     = True,       # Google email is verified
            )
            db.add(user)

    # Update last login
    user.last_login = datetime.now(timezone.utc)
    db.commit()
    db.refresh(user)

    if not user.is_active:
        return RedirectResponse(f"{FRONTEND_URL}/login?error=account_disabled")

    # Issue tokens
    jwt_access  = create_access_token(user.id, user.role)
    jwt_refresh = create_refresh_token(user.id)

    # Set refresh cookie
    redirect = RedirectResponse(
        url    = f"{FRONTEND_URL}/auth/callback?token={jwt_access}",
        status_code = 302,
    )
    redirect.set_cookie(
        key      = REFRESH_COOKIE_NAME,
        value    = jwt_refresh,
        httponly = True,
        secure   = False,
        samesite = "lax",
        max_age  = REFRESH_TOKEN_EXPIRE * 24 * 60 * 60,
        path     = "/api/auth",
    )

    return redirect
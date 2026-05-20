from pydantic import BaseModel, EmailStr, Field, field_validator
from datetime import datetime
from typing import Optional
from app.models.user import UserRole


# ── Register ───────────────────────────────────────────────────────────────
class RegisterRequest(BaseModel):
    email:    EmailStr
    username: str      = Field(..., min_length=3, max_length=30)
    password: str      = Field(..., min_length=8)

    @field_validator("username")
    @classmethod
    def username_alphanumeric(cls, v):
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Username can only contain letters, numbers, - and _")
        return v.lower()


# ── Login ──────────────────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    username_or_email: str
    password:          str


# ── Token response (access token only — refresh in cookie) ────────────────
class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"


# ── User response ──────────────────────────────────────────────────────────
class UserResponse(BaseModel):
    id:          int
    email:       str
    username:    str
    role:        UserRole
    is_active:   bool
    avatar_url:  Optional[str]
    created_at:  datetime
    last_login:  Optional[datetime]

    class Config:
        from_attributes = True


# ── Auth response (token + user) ───────────────────────────────────────────
class AuthResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    user:         UserResponse


# ── Feedback ───────────────────────────────────────────────────────────────
class FeedbackRequest(BaseModel):
    message: str  = Field(..., min_length=1, max_length=1000)
    rating:  Optional[float] = Field(None, ge=1, le=5)


class FeedbackResponse(BaseModel):
    id:         int
    message:    str
    rating:     Optional[float]
    created_at: datetime
    user:       UserResponse

    class Config:
        from_attributes = True
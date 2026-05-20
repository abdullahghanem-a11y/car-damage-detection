from sqlalchemy import Column, Integer, String, Boolean, DateTime, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum
from app.database import Base


class UserRole(str, enum.Enum):
    admin = "admin"
    user  = "user"


class User(Base):
    __tablename__ = "users"

    id               = Column(Integer, primary_key=True, index=True)
    email            = Column(String,  unique=True, nullable=False, index=True)
    username         = Column(String,  unique=True, nullable=False, index=True)
    hashed_password  = Column(String,  nullable=True)   # nullable for Google OAuth users
    google_id        = Column(String,  unique=True, nullable=True, index=True)
    avatar_url       = Column(String,  nullable=True)
    role             = Column(Enum(UserRole), default=UserRole.user, nullable=False)
    is_active        = Column(Boolean, default=True,  nullable=False)
    is_verified      = Column(Boolean, default=False, nullable=False)
    created_at       = Column(DateTime(timezone=True), server_default=func.now())
    updated_at       = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    last_login       = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    sessions  = relationship("Session",  back_populates="user", cascade="all, delete-orphan")
    feedbacks = relationship("Feedback", back_populates="user", cascade="all, delete-orphan")
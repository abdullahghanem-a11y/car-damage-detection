from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base


class ActivityLog(Base):
    __tablename__ = "activity_logs"

    id         = Column(Integer, primary_key=True, index=True)
    user_id    = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    action     = Column(String,  nullable=False, index=True)  # e.g. "login", "detect", "delete_session"
    method     = Column(String,  nullable=True)               # GET, POST, DELETE, etc.
    path       = Column(String,  nullable=True)               # /api/detect/
    details    = Column(Text,    nullable=True)               # JSON string with extra info
    ip_address = Column(String,  nullable=True)
    status     = Column(Integer, nullable=True)               # HTTP status code
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    user = relationship("User", foreign_keys=[user_id])
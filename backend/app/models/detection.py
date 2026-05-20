from sqlalchemy import Column, Integer, String, DateTime, JSON, Text, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base


class Session(Base):
    __tablename__ = "sessions"

    id               = Column(Integer, primary_key=True, index=True)
    user_id          = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name             = Column(String,  nullable=False)
    model_version    = Column(String,  nullable=False)
    total_instances  = Column(Integer, default=0)
    overall_severity = Column(String,  nullable=True)
    created_at       = Column(DateTime(timezone=True), server_default=func.now())
    updated_at       = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    user       = relationship("User",      back_populates="sessions")
    detections = relationship("Detection", back_populates="session", cascade="all, delete-orphan")


class Detection(Base):
    __tablename__ = "detections"

    id               = Column(Integer, primary_key=True, index=True)
    session_id       = Column(Integer, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    image_filename   = Column(String,  nullable=False)
    image_path       = Column(String,  nullable=False)
    annotated_image  = Column(Text,    nullable=True)
    results          = Column(JSON,    nullable=False, default=list)
    total_instances  = Column(Integer, default=0)
    model_version    = Column(String,  nullable=False)
    verified         = Column(Integer, default=1)
    vehicle_type     = Column(String,  nullable=True)
    overall_severity = Column(String,  nullable=True)
    created_at       = Column(DateTime(timezone=True), server_default=func.now())

    session = relationship("Session", back_populates="detections")
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.sql import func
from app.database import Base


class Detection(Base):
    __tablename__ = "detections"

    id               = Column(Integer, primary_key=True, index=True)
    image_filename   = Column(String, nullable=False)
    image_path       = Column(String, nullable=False)
    results          = Column(JSON, nullable=False)
    total_instances  = Column(Integer, default=0)
    model_version    = Column(String, nullable=False)
    verified         = Column(Integer, default=1)   # 1=verified, 0=rejected
    vehicle_type     = Column(String, nullable=True)
    overall_severity = Column(String, nullable=True) # Minor/Moderate/Severe
    created_at       = Column(DateTime(timezone=True), server_default=func.now())
# reset_db.py — run once from backend/ folder to reset the database
from app.database import engine, Base
from app.models.user      import User
from app.models.feedback  import Feedback
from app.models.detection import Session, Detection
from app.models.log       import ActivityLog

print("Dropping all tables...")
Base.metadata.drop_all(bind=engine)
print("Recreating all tables...")
Base.metadata.create_all(bind=engine)
print("Done ✅")
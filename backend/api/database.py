from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime
import os

# Create data directory if it doesn't exist
DB_DIR = "backend/data"
os.makedirs(DB_DIR, exist_ok=True)
DATABASE_URL = f"sqlite:///./{DB_DIR}/predictions.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, index=True)
    vaccine_type = Column(String)
    predicted_titer_28 = Column(Float)
    risk_tier = Column(String)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    # Store full input and full trajectory as JSON
    input_data = Column(JSON)
    full_output = Column(JSON)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

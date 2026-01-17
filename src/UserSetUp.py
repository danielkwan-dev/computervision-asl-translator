from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from datetime import datetime

engine = create_engine('sqlite:///asl_database.db', echo=False)
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    total_xp = Column(Integer, default=0)
    current_streak = Column(Integer, default=0)
    
    last_practice_date = Column(DateTime, default=datetime.now)
    
    progress = relationship("UserProgress", back_populates="user")

class UserProgress(Base):
    __tablename__ = 'user_progress'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    letter = Column(String)
    mastery_score = Column(Integer, default=0)
    
    user = relationship("User", back_populates="progress")

# Create the tables if they don't exist
def init_db():
    Base.metadata.create_all(engine)
    print("Database initialized successfully.")

if __name__ == "__main__":
    init_db()
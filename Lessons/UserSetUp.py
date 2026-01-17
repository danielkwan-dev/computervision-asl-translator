from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, relationship
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

class Lesson(Base):
    __tablename__ = 'lessons'
    
    id = Column(Integer, primary_key=True)
    title = Column(String)          # e.g., "The Basics"
    target_letters = Column(String) # e.g., "A,B,C,D,E"
    difficulty_level = Column(Integer)

class UserProgress(Base):
    __tablename__ = 'user_progress'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    letter = Column(String)         # e.g., "A"
    mastery_score = Column(Integer, default=0)
    
    user = relationship("User", back_populates="progress")

def init_db():
    Base.metadata.create_all(engine)
    print("Database initialized successfully.")

if __name__ == "__main__":
    init_db()
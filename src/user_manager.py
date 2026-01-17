from sqlalchemy.orm import sessionmaker
from UserSetUp import engine, User, UserProgress, init_db
from datetime import datetime, timedelta

Session = sessionmaker(bind=engine)
session = Session()

def create_user(username):
    """Creates a new user if one doesn't exist."""
    
    # Check if a user already exists
    existing = session.query(User).filter_by(username=username).first()
    if existing:
        print(f"User '{username}' already exists (ID: {existing.id}).")
        return existing

    # Create new user.
    new_user = User(username=username)
    session.add(new_user)
    session.commit()
    print(f"Created new user: {username}")
    return new_user

def get_user(username):
    """ Retrieve user by username."""

    return session.query(User).filter_by(username=username).first()

def update_streak(user):
    """ Updates the user's practice streak based on last practice date. """

    now = datetime.now()
    last = user.last_practice_date
    
    diff_days = (now.date() - last.date()).days

    if diff_days == 0:
        pass
    elif diff_days == 1:
        user.current_streak += 1
    else:
        user.current_streak = 1
    
    user.last_practice_date = now

def record_attempt(username, letter, is_correct):
    """ Records a user's attempt at a letter and updates XP, streak, and mastery. """

    user = get_user(username)
    if not user:
        print("User not found!")
        return

    # Find existing progress, if there is none, create new
    progress = session.query(UserProgress).filter_by(user_id=user.id, letter=letter).first()
    
    if not progress:
        progress = UserProgress(user_id=user.id, letter=letter, mastery_score=0)
        session.add(progress)
    
    if is_correct:
        xp_gain = 10
        user.total_xp += xp_gain
        
        update_streak(user)
        
        progress.mastery_score = min(100, progress.mastery_score + 10)
        
        print(f"Correct! +{xp_gain} XP | Streak: {user.current_streak} | '{letter}' Mastery: {progress.mastery_score}%")
    else:
        progress.mastery_score = max(0, progress.mastery_score - 5)
        print(f"Incorrect. '{letter}' Mastery: {progress.mastery_score}%")

    session.commit()

# --- TEST ---
if __name__ == "__main__":
    init_db()
    me = create_user("Student")
    
    print("\n--- Attempt 1: Correct 'A' ---")
    record_attempt("WaterlooStudent", "A", True)

    print("\n--- Attempt 2: Incorrect 'B' ---")
    record_attempt("WaterlooStudent", "B", False)

    refreshed_user = get_user("WaterlooStudent")
    print(f"\nFINAL STATS: XP={refreshed_user.total_xp}, Streak={refreshed_user.current_streak}")
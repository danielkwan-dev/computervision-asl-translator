# Real-Time ASL Recognition Engine

Real-time American Sign Language (ASL) letter recognition using hand landmark detection and machine learning.

Demo: 

## Features

- Real-time hand tracking with MediaPipe
- ASL alphabet recognition (A-Z, plus delete, space, nothing)
- Structured lessons with a learning path ranging from "The Basics" to "Advanced"
- User Profiles: Create accounts, save progress, track statistics via a local database using SQL
- The app automatically detects which letters you struggle with and focuses on them
- Earn XP for correct signs and maintain daily streaks to stay motivated
- Hold-to-confirm system for accurate letter input
- Build words and sentences by signing letters

## Setup

### Requirements

- Python 3.10+
- Webcam

### Installation

```bash
git clone https://github.com/JummyJoeJackson/handtrack.git
cd handtrack
pip install -r requirements.txt
```

### Dependencies

- opencv-python
- mediapipe
- numpy
- pandas
- torch
- scikit-learn
- SQLAlchemy

## Usage

### Run the Application

```bash
python main.py
```
## Run Simple Inference (No Game)

```bash
python inference.py --model asl_model.pth
```

### Controls

- **Hold a sign** for 2 seconds to confirm the letter
- **Press `c`** (in inference mode) to clear text
- **Press `q`** to quit

### Options

```bash
python src/inference.py --model models/asl_model.pth --hold-time 1.5 --camera 0
```

- `--hold-time`: Seconds to hold pose before confirming (default: 2.0)
- `--camera`: Camera ID if you have multiple webcams (default: 0)

## How It Works

1. **Hand Detection**: MediaPipe detects 21 hand landmarks from webcam feed
2. **Normalization**: Landmarks are centered and scaled for consistency
3. **Classification**: Neural network predicts the ASL letter
4. **Hold-to-Confirm**: Hold a sign steady to add it to your text

## Supported Signs

- Letters: A-Y (Not including J, Z yet)
- Special: Space, Delete, Nothing

## Training

### Collect Custom Data
python main.py
Press letter keys (a-z) while showing hand signs to capture samples.

## Project Structure

```
handtrack/
├── main.py                 # Main Game Application & Menu
├── user_manager.py         # User logic, stats, and database management
├── UserSetUp.py            # Database models (SQLAlchemy)
├── create_lessons.py       # Script to populate the curriculum
├── asl_database.db         # Local SQLite database (generated on run)
├── hand_landmarker.task    # MediaPipe hand model
├── requirements.txt
├── asl_model.pth           # Pre-trained model
└── src/
    ├── model.py            # Neural network architecture
    ├── train.py            # Model training
    ├── inference.py        # Standalone recognition app
    └── convert_images.py   # Dataset conversion utility
```

## Technologies Used

- **MediaPipe** - Hand landmark detection (Tasks API)
- **PyTorch** - Neural network framework (Multilayer Perceptron)
- **OpenCV** - Video capture and display
- SQLAlchemy - Database management for user profiles
- **Kaggle ASL Alphabet Dataset** - Training data source

## Known Limitations

- **Letters J and Z**: These letters require hand movement to form (drawing a "J" or "Z" shape in the air). Since this system uses static hand poses, J and Z cannot be reliably recognized.

## License

MIT

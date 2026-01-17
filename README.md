# ASL Hand Tracking

Real-time American Sign Language (ASL) letter recognition using hand landmark detection and machine learning.

## Features

- Real-time hand tracking with MediaPipe
- ASL alphabet recognition (A-Z, plus delete, space, nothing)
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

## Usage

### Run the Application

```bash
python src/inference.py --model models/asl_model.pth
```

### Controls

- **Hold a sign** for 2 seconds to confirm the letter
- **Press `c`** to clear text
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

- Letters: A-Z
- Special: Space, Delete, Nothing

## Project Structure

```
handtrack/
├── main.py                 # Data collection utility
├── hand_landmarker.task    # MediaPipe hand model
├── requirements.txt
├── src/
│   ├── model.py            # Neural network architecture
│   ├── train.py            # Model training
│   ├── inference.py        # Real-time recognition app
│   └── convert_images.py   # Dataset conversion utility
└── models/
    └── asl_model.pth       # Pre-trained model
```

## Technologies Used

- **MediaPipe** - Hand landmark detection (Tasks API)
- **PyTorch** - Neural network framework
- **OpenCV** - Video capture and display
- **Kaggle ASL Alphabet Dataset** - Training data source

## License

MIT

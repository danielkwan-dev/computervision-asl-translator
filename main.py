import cv2
import mediapipe as mp
import time
import torch
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- IMPORTS FROM YOUR TEAM ---
import user_manager             # Your Logic
from model import get_model, normalize_landmarks, ASL_CLASSES  # Friend's Model Logic

# --- CONFIGURATION ---
MODEL_PATH = "hand_landmarker.task"   # MediaPipe Model
TRAINED_MODEL_PATH = "asl_model.pth"  # Friend's PyTorch Model (Output of train.py)
CONFIDENCE_THRESHOLD = 0.7
TARGET_HOLD_TIME = 2.0                # Seconds to hold sign for success

# --- THE BRIDGE CLASS ---
class ASLInferenceBridge:
    """
    Connects MediaPipe (Your code) to PyTorch (Friend's code).
    """
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading AI Model on: {self.device}")
        
        try:
            # 1. Load the file
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 2. Extract metadata
            # If the file has 'num_classes', use it. Otherwise default to 29.
            num_classes = checkpoint.get('num_classes', len(ASL_CLASSES))
            
            # 3. Initialize the model structure
            self.model = get_model(model_type="mlp", num_classes=num_classes)
            
            # 4. CRITICAL FIX: Load the weights from the 'model_state_dict' key
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # Fallback for older model versions that might be just weights
                self.model.load_state_dict(checkpoint)
                
            self.model.to(self.device)
            self.model.eval()
            
            # 5. Load dynamic labels if available (safer than hardcoding)
            if 'idx_to_label' in checkpoint:
                self.idx_to_label = checkpoint['idx_to_label']
            else:
                self.idx_to_label = {i: c for i, c in enumerate(ASL_CLASSES)}
                
            print("✅ AI Model loaded successfully!")
            
        except FileNotFoundError:
            print(f"❌ ERROR: Could not find '{model_path}'.")
            print("   Run 'python train.py' first to generate it.")
            self.model = None

    def predict(self, landmarks_list):
        if not self.model: return "?"

        # 1. Convert to Numpy
        landmarks_np = np.array(landmarks_list, dtype=np.float32)
        
        # 2. Normalize
        norm_landmarks = normalize_landmarks(landmarks_np)
        
        # 3. Convert to PyTorch Tensor
        input_tensor = torch.tensor(norm_landmarks, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # 4. Run Inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
        
        # 5. Return label using the loaded map
        if confidence.item() > 0.6: 
            idx = predicted_idx.item()
            # Use the dictionary loaded from the file, or fallback to list
            if isinstance(self.idx_to_label, dict):
                return self.idx_to_label.get(idx, "?")
            else:
                return ASL_CLASSES[idx]
        return "?"

# --- MAIN APP LOOP ---
def main():
    # 1. SETUP
    print("--- LAUNCHING ASL TRAINER ---")
    
    # Login
    current_user = user_manager.login()
    if not current_user: return

    # Load AI
    ai_brain = ASLInferenceBridge(TRAINED_MODEL_PATH)

    # Load Camera & MediaPipe
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1
    )
    landmarker = vision.HandLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(0)

    # 2. GAME STATE INITIALIZATION
    current_lesson = user_manager.get_next_lesson(current_user)
    if not current_lesson:
        print("All lessons complete!")
        return

    stats = user_manager.get_lesson_status(current_user, current_lesson)
    target_letter = min(stats, key=stats.get) # Pick weakest letter
    
    hold_start_time = None
    
    print(f"\nGenerative Lesson: {current_lesson.title}")
    print(f"TASK: Sign the letter '{target_letter}'")

    # 3. VIDEO LOOP
    while True:
        success, frame = cap.read()
        if not success: break
        
        # Mirror the frame for natural feeling
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect Hands
        detection_result = landmarker.detect(mp_image)
        
        predicted_char = "?"
        
        if detection_result.hand_landmarks:
            hand_landmarks = detection_result.hand_landmarks[0]
            
            # --- EXTRACT LANDMARKS ---
            # We must flatten the data: [x0, y0, z0, x1, y1, z1, ...]
            # This matches the input format of 'asl_data.csv'
            raw_landmarks = []
            height, width, _ = frame.shape
            
            for lm in hand_landmarks:
                raw_landmarks.extend([lm.x, lm.y, lm.z])
                
                # Draw landmarks (Visual helper)
                cx, cy = int(lm.x * width), int(lm.y * height)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

            # --- PREDICT ---
            predicted_char = ai_brain.predict(raw_landmarks)

        # --- GAME LOGIC ---
        color = (0, 0, 255) # Red
        status_msg = f"Target: {target_letter} | You: {predicted_char}"

        if predicted_char == target_letter:
            color = (0, 255, 0) # Green
            
            if hold_start_time is None:
                hold_start_time = time.time()
            
            elapsed = time.time() - hold_start_time
            progress = min(1.0, elapsed / TARGET_HOLD_TIME)
            
            # Draw Progress Bar
            cv2.rectangle(frame, (50, 200), (50 + int(200 * progress), 220), (0, 255, 0), -1)
            
            if elapsed >= TARGET_HOLD_TIME:
                # SUCCESS!
                user_manager.record_attempt(current_user.username, target_letter, True)
                
                # Get next target
                current_lesson = user_manager.get_next_lesson(current_user)
                if current_lesson:
                    stats = user_manager.get_lesson_status(current_user, current_lesson)
                    target_letter = min(stats, key=stats.get)
                    hold_start_time = None
                else:
                    status_msg = "LESSON COMPLETE!"
        else:
            hold_start_time = None

        # --- DRAW UI ---
        # Top Bar
        cv2.rectangle(frame, (0, 0), (640, 60), (0, 0, 0), -1)
        cv2.putText(frame, status_msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"XP: {current_user.total_xp} | Streak: {current_user.current_streak}", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("ASL Trainer", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
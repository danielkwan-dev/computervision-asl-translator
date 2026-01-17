import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


PATH = "hand_landmarker.task"
IMAGE_PATH = "asl_alphabet_test/B_test.jpg"


def extract_landmarks(image_path):
    landmarks_list = []
    
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.1,
        min_hand_presence_confidence=0.1
    )

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        cv_img = cv2.imread(image_path)
        if cv_img is None:
            return []
            
        cv_img = cv2.copyMakeBorder(cv_img, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        cv_img = cv2.convertScaleAbs(cv_img, alpha=1.3, beta=10)
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
        
        mp_results = landmarker.detect(mp_image)

        if mp_results.hand_landmarks:
            width, height = mp_image.width, mp_image.height
            for hand_landmarks in mp_results.hand_landmarks:
                for landmark in hand_landmarks:
                    px = int(landmark.x * width)
                    py = int(landmark.y * height)
                    pz = int(landmark.z * width)
                    landmarks_list.append([px - 50, py - 50, pz])
    
    return landmarks_list


if __name__ == "__main__":
    landmarks = extract_landmarks(IMAGE_PATH)

    print(f"Extracted {len(landmarks)} landmarks from {IMAGE_PATH}")
    if landmarks:
        print("Landmarks list:")
        print(landmarks)

        image = cv2.imread(IMAGE_PATH)
        if image is not None:
            for landmark in landmarks:
                cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 255, 0), -1)
            
            cv2.imshow("ASL Landmark Detection", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print(f"Could not load image for visualization: {IMAGE_PATH}")

import os
import cv2
import glob
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import dlib
import logging
import config
from check_user import FaceRecognizer

def evaluate_model(recognizer, test_dir=config.KNOWN_FACES_DIR, threshold=0.4):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(config.SHAPE_PREDICTOR)
    face_reco_model = dlib.face_recognition_model_v1(config.FACEREC_MODEL)

    y_true = []
    y_pred = []

    for person_folder in os.listdir(test_dir):
        person_path = os.path.join(test_dir, person_folder)
        if not os.path.isdir(person_path):
            continue

        for img_path in glob.glob(os.path.join(person_path, "*.jpg")):
            img = cv2.imread(img_path)
            if img is None:
                continue
            faces = detector(img, 1)
            if len(faces) == 0:
                continue
            shape = predictor(img, faces[0])
            face_feature = face_reco_model.compute_face_descriptor(img, shape)

            dists = [recognizer.return_euclidean_distance(face_feature, known_feat) 
                     for known_feat in recognizer.face_features_known_list]
            min_dist = min(dists)
            if min_dist < threshold:
                pred_name = recognizer.face_name_known_list[dists.index(min_dist)]
            else:
                pred_name = "unknown"

            true_name = person_folder.split("_", 1)[-1]
            y_true.append(true_name)
            y_pred.append(pred_name)

    acc = accuracy_score(y_true, y_pred)
    print("Accuracy:", round(acc * 100, 2), "%")
    print("Classification Report:\n", classification_report(y_true, y_pred))

def main():
    logging.basicConfig(level=logging.INFO)
    recognizer = FaceRecognizer()
    recognizer.get_face_database()   # loads face features
    evaluate_model(recognizer)       # evaluates accuracy
    recognizer.run()                 # only if you want real-time recognition afterwards

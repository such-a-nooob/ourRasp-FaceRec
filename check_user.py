import dlib
import numpy as np
import cv2
import os
import pandas as pd 
import time
import logging
import sqlite3
import datetime
import features_extraction_to_csv as fe
import config

# Dlib setup
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(config.SHAPE_PREDICTOR)
face_reco_model = dlib.face_recognition_model_v1(config.FACEREC_MODEL)

# Ensure table exists in the found.db database
conn = sqlite3.connect(config.FOUND_DB_PATH)
cursor = conn.cursor()
cursor.execute(f"""
CREATE TABLE IF NOT EXISTS {config.TABLE_NAME} (
    name TEXT,
    time TEXT,
    date DATE,
    image_path TEXT,
    UNIQUE(time)
)
""")
conn.commit()
conn.close()


class FaceRecognizer:
    def __init__(self):
        self.font = cv2.FONT_ITALIC
        self.frame_cnt = 0
        self.fps = 0
        self.start_time = time.time()

        self.face_features_known_list = []
        self.face_name_known_list = []
        self.last_frame_face_name_list = []
        self.current_frame_face_name_list = []
        self.current_frame_face_centroid_list = []
        self.current_frame_face_cnt = 0
        self.last_frame_face_cnt = 0
        self.reclassify_interval = 10
        self.reclassify_interval_cnt = 0

        self.detected_faces_dir = config.DETECTED_FACES_DIR
        os.makedirs(self.detected_faces_dir, exist_ok=True)
        self.last_saved_time = {}

    def get_face_database(self):
        fe.main()
        if os.path.exists(config.FEATURES_CSV_PATH):
            csv_rd = pd.read_csv(config.FEATURES_CSV_PATH, header=None)
            for i in range(csv_rd.shape[0]):
                features_someone_arr = [csv_rd.iloc[i][j] if csv_rd.iloc[i][j] != "" else "0" for j in range(1, 129)]
                self.face_name_known_list.append(csv_rd.iloc[i][0])
                self.face_features_known_list.append(features_someone_arr)
            logging.info("Faces in Database: %d", len(self.face_features_known_list))
            return True
        else:
            logging.warning("Feature file not found!")
            return False

    @staticmethod
    def return_euclidean_distance(feature_1, feature_2):
        return np.linalg.norm(np.array(feature_1) - np.array(feature_2))

    def found(self, name, img_rd, face_position):
        current_time = datetime.datetime.now()
        current_time_str = current_time.strftime("%H:%M:%S")
        current_date = current_time.strftime("%Y-%m-%d")

        if name in self.last_saved_time:
            time_diff = (current_time - self.last_saved_time[name]).total_seconds() / 60
            if time_diff < 5:
                print(f"{name} already saved within the last 5 minutes. Skipping.")
                return

        date_folder = os.path.join(self.detected_faces_dir, current_date)
        os.makedirs(date_folder, exist_ok=True)

        face_image_path = os.path.join(date_folder, f"{name}_{current_time_str.replace(':', '-')}.jpg")
        x, y, w, h = face_position
        face_img = img_rd[y:y + h, x:x + w]
        cv2.imwrite(face_image_path, face_img)

        conn = sqlite3.connect(config.FOUND_DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            f"INSERT OR IGNORE INTO {config.TABLE_NAME} (name, time, date, image_path) VALUES (?, ?, ?, ?)",
            (name, current_time_str, current_date, face_image_path),
        )
        conn.commit()
        conn.close()

        self.last_saved_time[name] = current_time
        print(f"{name} marked present at {current_time_str} on {current_date}. Image saved at {face_image_path}")

    def run(self):
        if not self.get_face_database():
            return

        cap = cv2.VideoCapture(config.CAMERA_INDEX)
        while cap.isOpened():
            self.frame_cnt += 1
            ret, img_rd = cap.read()
            if not ret:
                break

            faces = detector(img_rd, 0)
            self.current_frame_face_cnt = len(faces)

            if self.current_frame_face_cnt == 0:
                self.current_frame_face_name_list = []
            else:
                self.current_frame_face_name_list = []
                for i in range(len(faces)):
                    shape = predictor(img_rd, faces[i])
                    face_feature = face_reco_model.compute_face_descriptor(img_rd, shape)
                    e_distances = [self.return_euclidean_distance(face_feature, feat) for feat in self.face_features_known_list]
                    min_dist = min(e_distances)
                    match_idx = e_distances.index(min_dist)

                    if min_dist < 0.4:
                        name = self.face_name_known_list[match_idx]
                        self.current_frame_face_name_list.append(name)
                        face_position = (
                            faces[i].left(),
                            faces[i].top(),
                            faces[i].right() - faces[i].left(),
                            faces[i].bottom() - faces[i].top(),
                        )
                        self.found(name, img_rd, face_position)
                    else:
                        self.current_frame_face_name_list.append("unknown")

            for i, d in enumerate(faces):
                cv2.rectangle(img_rd, (d.left(), d.top()), (d.right(), d.bottom()), (0, 255, 0), 1)
                cv2.putText(img_rd, self.current_frame_face_name_list[i], (d.left(), d.top() - 10), self.font, 0.6, (0, 255, 0), 1)

            
            cv2.putText(
                img_rd, "Press 'x' to exit", 
                (10, img_rd.shape[0] - 10),  # bottom-left corner
                self.font, 0.5, (0, 255, 255), 1
            )

            cv2.imshow("Face Recognition", img_rd)
            if cv2.waitKey(1) & 0xFF == ord('x'):
                break

        cap.release()
        cv2.destroyAllWindows()


def main():
    logging.basicConfig(level=logging.INFO)
    recognizer = FaceRecognizer()
    recognizer.run()


if __name__ == "__main__":
    main()

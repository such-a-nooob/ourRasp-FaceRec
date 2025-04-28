import dlib
import numpy as np
import cv2
import os
import pandas as pd
import time
import logging
import sqlite3
import datetime
import shutil
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

        self.recognized_faces_dir = config.RECOGNIZED_FACES_DIR
        os.makedirs(self.recognized_faces_dir, exist_ok=True)
        self.last_saved_time = {}

    def get_face_database(self):
        logging.info("Running feature extraction...")
        fe.main()  # keep this as requested
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

    def found(self, name, img_rd, face_position, source_img_path):
        try:
            # Extract date and time from the source image filename
            basename = os.path.basename(source_img_path)  # e.g., face_20250428_155157.jpg
            parts = basename.split('_')
            if len(parts) < 3:
                raise ValueError(f"Invalid filename format: {basename}")

            date_raw = parts[1]  # '20250428'
            time_raw = parts[2].split('.')[0]  # '155157'

            # Format extracted date and time
            current_date = f"{date_raw[:4]}-{date_raw[4:6]}-{date_raw[6:]}"  # '2025-04-28'
            current_time_str = f"{time_raw[:2]}:{time_raw[2:4]}:{time_raw[4:]}"  # '15:51:57'

            # Create a datetime object from extracted date and time
            extracted_datetime = datetime.datetime(
                int(date_raw[:4]), int(date_raw[4:6]), int(date_raw[6:]),
                int(time_raw[:2]), int(time_raw[2:4]), int(time_raw[4:])
            )

        except Exception as e:
            logging.error(f"Error extracting date and time from filename: {e}")
            return

        # Check last saved time to avoid duplicate saves
        if name in self.last_saved_time:
            time_diff = (extracted_datetime - self.last_saved_time[name]).total_seconds() / 60
            if time_diff < 5:
                logging.info(f"{name} already saved within the last 5 minutes. Skipping.")
                return

        # Save cropped face to recognized_faces_dir
        date_folder = os.path.join(self.recognized_faces_dir, current_date)
        os.makedirs(date_folder, exist_ok=True)

        face_image_filename = f"{name}_{current_time_str.replace(':', '-')}.jpg"
        face_image_path = os.path.join(date_folder, face_image_filename)
        x, y, w, h = face_position
        face_img = img_rd[y:y + h, x:x + w]
        cv2.imwrite(face_image_path, face_img)

        # Log entry in DB
        conn = sqlite3.connect(config.FOUND_DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            f"INSERT OR IGNORE INTO {config.TABLE_NAME} (name, time, date, image_path) VALUES (?, ?, ?, ?)",
            (name, current_time_str, current_date, face_image_path),
        )
        conn.commit()
        conn.close()

        # Copy full original image to static/recognized_faces/{yyyy-mm-dd} directory
        static_dest_dir = os.path.join("static", "recognized_faces", current_date)
        os.makedirs(static_dest_dir, exist_ok=True)
        full_image_filename = f"{name}_{current_time_str.replace(':', '-')}.jpg"
        full_image_dest_path = os.path.join(static_dest_dir, full_image_filename)
        shutil.copy2(source_img_path, full_image_dest_path)

        # Update last_saved_time with extracted timestamp
        self.last_saved_time[name] = extracted_datetime

        logging.info(f"{name} marked at {current_time_str} on {current_date}.")

    def run(self):
        if not self.get_face_database():
            return

        current_time = datetime.datetime.now()
        current_date = current_time.strftime("%Y-%m-%d")
        all_faces_dir = config.ALL_FACES_DIR
        image_dir = os.path.join(all_faces_dir, current_date)
        logging.info(f"Processing images from directory: {image_dir}")

        image_files = []
        for root, _, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(os.path.join(root, file))

        if not image_files:
            logging.warning("No images found in the directory.")
            return

        for img_path in image_files:
            self.frame_cnt += 1
            #logging.info(f"Processing image: {img_path}")
            img_rd = cv2.imread(img_path)

            if img_rd is None:
                logging.warning(f"Failed to load image: {img_path}")
                continue

            faces = detector(img_rd, 0)
            self.current_frame_face_cnt = len(faces)
            #logging.info(f"Detected {self.current_frame_face_cnt} face(s) in {img_path}")

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
                        self.found(name, img_rd, face_position, img_path)
                    else:
                        self.current_frame_face_name_list.append("unknown")

        logging.info("Processing complete. All faces scanned.")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    recognizer = FaceRecognizer()
    recognizer.run()


if __name__ == "__main__":
    main()

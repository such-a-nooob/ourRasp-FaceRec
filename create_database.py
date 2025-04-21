import dlib
import numpy as np
import cv2
import os
import time
import logging
import tkinter as tk
from tkinter import font as tkFont
from tkinter import LabelFrame
from PIL import Image, ImageTk
import config

detector = dlib.get_frontal_face_detector()


class Face_Register:
    def __init__(self):
        self.current_frame_faces_cnt = 0
        self.existing_faces_cnt = 0
        self.ss_cnt = 0
        self.current_face_dir = ""
        self.input_name_char = ""
        self.face_folder_created_flag = False
        self.out_of_range_flag = False

        self.win = tk.Tk()
        self.win.title("Face Register")
        self.win.geometry("1100x560")

        self.font = tkFont.Font(family='Arial', size=12)
        self.label_font = tkFont.Font(family='Arial', size=10)

        # Camera frame
        self.frame_camera = LabelFrame(self.win, text="Camera Feed", font=self.label_font, padx=5, pady=5)
        self.frame_camera.grid(row=0, column=0, padx=10, pady=10)
        self.label_img = tk.Label(self.frame_camera)
        self.label_img.pack()

        # Info + Control frame
        self.frame_info = LabelFrame(self.win, text="Controls & Info", font=self.label_font, padx=10, pady=10)
        self.frame_info.grid(row=0, column=1, sticky="n", padx=10, pady=10)
        self.frame_info.config(width=380, height=512)
        self.frame_info.grid_propagate(False)

        self.label_fps = tk.Label(self.frame_info, text="FPS: 0", font=self.label_font, fg="darkgreen")
        self.label_fps.pack(anchor="w")

        self.label_cnt_face = tk.Label(self.frame_info, text="Faces in current frame: 0", font=self.label_font)
        self.label_cnt_face.pack(anchor="w")

        self.label_cnt_db = tk.Label(self.frame_info, text="Faces in database: 0", font=self.label_font)
        self.label_cnt_db.pack(anchor="w")

        self.label_warning = tk.Label(self.frame_info, text="", font=self.label_font, fg='red')
        self.label_warning.pack(anchor="w", pady=(5, 10))

        # Name input
        self.frame_name = LabelFrame(self.frame_info, text="Enter Name", font=self.label_font, padx=5, pady=5)
        self.frame_name.pack(fill="x", pady=(5, 5))
        self.entry_name = tk.Entry(self.frame_name, font=self.label_font)
        self.entry_name.pack(fill="x", pady=(0, 5))

        self.btn_create_folder = tk.Button(self.frame_name, text="Create Folder", font=self.label_font,
                                           command=self.get_input_name)
        self.btn_create_folder.pack()

        self.btn_add_face = tk.Button(self.frame_info, text="Add Face", font=self.label_font,
                                      command=self.save_current_face)
        self.btn_add_face.pack(fill="x", pady=(10, 5))

        self.label_log = tk.Label(self.frame_info, text="", wraplength=320, justify="left", font=self.label_font)
        self.label_log.pack(anchor="w", pady=(10, 0))

        self.cap = cv2.VideoCapture(config.CAMERA_INDEX)
        self.frame_start_time = time.time()

        if not os.path.exists(config.KNOWN_FACES_DIR):
            os.makedirs(config.KNOWN_FACES_DIR)

        for person in os.listdir(config.KNOWN_FACES_DIR):
            self.existing_faces_cnt += 1
        self.label_cnt_db.config(text="Faces in database: " + str(self.existing_faces_cnt))

        self.current_frame = None
        self.face_ROI_image = None

        self.process()
        self.win.mainloop()

    def get_input_name(self):
        self.input_name_char = self.entry_name.get()
        if self.input_name_char:
            self.existing_faces_cnt += 1
            self.current_face_dir = os.path.join(config.KNOWN_FACES_DIR, "person" + str(self.existing_faces_cnt) + "_" + self.input_name_char)
            os.makedirs(self.current_face_dir)
            self.face_folder_created_flag = True
            self.label_log.config(text="Created folder: " + self.current_face_dir)
            self.label_cnt_db.config(text="Faces in database: " + str(self.existing_faces_cnt))

    def save_current_face(self):
        if not self.face_folder_created_flag:
            self.label_log.config(text="Please input name and create folder first!")
            return

        if self.current_frame_faces_cnt != 1:
            self.label_log.config(text="Only one face should be in the frame!")
            return

        if self.face_ROI_image is not None:
            self.ss_cnt += 1
            filename = os.path.join(self.current_face_dir, f"face{self.ss_cnt}.jpg")
            cv2.imwrite(filename, self.face_ROI_image)
            self.label_log.config(text=f"Saved face image as {filename}")

    def update_fps(self):
        now = time.time()
        fps = 1.0 / (now - self.frame_start_time)
        self.frame_start_time = now
        self.label_fps.config(text=f"FPS: {fps:.2f}")

    def process(self):
        ret, frame = self.cap.read()
        if not ret:
            self.label_log.config(text="Camera not found!")
            return

        frame = cv2.resize(frame, (640, 480))
        self.current_frame = frame.copy()
        faces = detector(frame, 0)

        self.update_fps()
        self.label_cnt_face.config(text=f"Faces in current frame: {len(faces)}")
        self.current_frame_faces_cnt = len(faces)

        for d in faces:
            x1 = max(d.left() - int(d.width() / 2), 0)
            y1 = max(d.top() - int(d.height() / 2), 0)
            x2 = min(d.right() + int(d.width() / 2), frame.shape[1])
            y2 = min(d.bottom() + int(d.height() / 2), frame.shape[0])

            if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                self.out_of_range_flag = True
                self.label_warning.config(text="Face out of range")
                color = (0, 0, 255)
            else:
                self.out_of_range_flag = False
                self.label_warning.config(text="")
                color = (255, 255, 255)
                self.face_ROI_image = frame[y1:y2, x1:x2]

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.label_img.imgtk = imgtk
        self.label_img.configure(image=imgtk)

        self.win.after(10, self.process)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    Face_Register()

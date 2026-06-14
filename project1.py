# app.py
from flask import Flask, render_template, request, redirect, url_for, Response
import os
import cv2
import numpy as np
import face_recognition
import time

app = Flask(__name__)
DATA_DIR = "data"

# Store loaded encodings in memory
known_face_encodings = []
known_face_names = []

# Ensure data folder exists
def create_data_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

# Load known faces from saved txt files
def load_known_faces():
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []

    for file_name in os.listdir(DATA_DIR):
        if file_name.endswith(".txt"):
            with open(os.path.join(DATA_DIR, file_name), "r") as file:
                details = file.readlines()
                if len(details) >= 5:
                    try:
                        face_encoding_str = details[4].split(":")[1].strip()
                        face_encoding = np.fromstring(face_encoding_str, sep=',')
                        known_face_encodings.append(face_encoding)
                        known_face_names.append(details[0].split(":")[1].strip())
                    except Exception as e:
                        print(f"Error loading {file_name}: {e}")

# Extract encoding from uploaded image
def extract_face_encodings(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return [], []
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    return face_locations, face_encodings

# Draw detections on frame
def detect_faces(frame):
    rgb_small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        details_line1 = "Unknown"
        details_line2 = ""

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

            # Load details from corresponding txt file
            txt_file = os.path.join(DATA_DIR, f"{name}.txt")
            if os.path.exists(txt_file):
                with open(txt_file, "r") as file:
                    details = file.readlines()
                    if len(details) >= 3:
                        details_line1 = details[0].strip()  # Name
                        details_line2 = f"{details[1].strip()} | {details[2].strip()}"  # Location + Phone

        # Resize coords back to original
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Color selection
        if name == "Unknown":
            box_color = (255, 0, 255)  # Pink for unknown
        else:
            box_color = (0, 255, 0)    # Green for known

        text_color = (255, 0, 0)  # Blue text (BGR)

        # Draw box
        cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
        # Background for text
        cv2.rectangle(frame, (left, bottom - 50), (right, bottom), box_color, cv2.FILLED)

        # Write text lines in blue
        cv2.putText(frame, details_line1, (left + 6, bottom - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 1)
        cv2.putText(frame, details_line2, (left + 6, bottom - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 1)

    return frame

# Camera frame generator
def gen_frames():
    camera = cv2.VideoCapture(0)
    prev_time = 0
    fps_limit = 10

    while True:
        success, frame = camera.read()
        if not success:
            break

        curr_time = time.time()
        if (curr_time - prev_time) > 1.0 / fps_limit:
            prev_time = curr_time
            frame = detect_faces(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        location = request.form['location']
        phone_number = request.form['phone_number']
        image = request.files['image']

        image_filename = f"{name}.jpg"
        image_path = os.path.join(DATA_DIR, image_filename)
        image.save(image_path)

        face_locations, face_encodings = extract_face_encodings(image_path)
        if face_encodings:
            with open(os.path.join(DATA_DIR, f"{name}.txt"), "w") as file:
                file.write(f"Name: {name}\n")
                file.write(f"Location: {location}\n")
                file.write(f"Phone Number: {phone_number}\n")
                file.write(f"Image: {image_filename}\n")
                file.write(f"Face Encoding: {','.join(str(x) for x in face_encodings[0])}\n")

            load_known_faces()

        return redirect(url_for('home'))
    return render_template('register.html')

@app.route('/find_missing_person')
def find_missing_person():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    create_data_dir()
    load_known_faces()
    app.run()

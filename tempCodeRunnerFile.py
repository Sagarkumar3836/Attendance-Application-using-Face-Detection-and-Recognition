import face_recognition
import pandas as pd
import cv2
import numpy as np
import os
import pickle
from flask import Flask, request, jsonify, send_file, render_template

app = Flask(__name__)

# Ensure the uploads directory exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load face encodings from the pickle file
with open('face_encodings.pickle', 'rb') as f:
    student_encodings = pickle.load(f)

def recognize_faces(input_image_path):
    # Load the input image
    input_image = face_recognition.load_image_file(input_image_path)
    
    # Detect faces in the input image
    face_locations = face_recognition.face_locations(input_image)
    face_encodings = face_recognition.face_encodings(input_image, face_locations)
    
    # If no faces are detected, return an empty result
    if not face_locations:
        return input_image, []
    
    # Convert the image to BGR for OpenCV
    input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    
    # Initialize results
    results = []
    confidence_threshold = 0.6  # Set your confidence threshold

    # Loop over each detected face
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        best_match_index = -1
        best_match_distance = float('inf')
        matched_roll_number = "Unknown"

        # Loop over each student encoding
        for roll_number, student_data in student_encodings.items():
            matches = face_recognition.compare_faces(student_data['encodings'], face_encoding)
            face_distances = face_recognition.face_distance(student_data['encodings'], face_encoding)
            
            # Find the best match for the detected face
            if len(face_distances) > 0:
                min_distance = np.min(face_distances)
                if min_distance < best_match_distance and min_distance < confidence_threshold:
                    best_match_index = np.argmin(face_distances)
                    best_match_distance = min_distance
                    matched_roll_number = roll_number

        # Check if a match was found
        if matched_roll_number != "Unknown":
            results.append((top, right, bottom, left, matched_roll_number))
            
            # Draw a box around the face
            cv2.rectangle(input_image, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Draw a label with the roll number
            label = f"{matched_roll_number}"
            cv2.putText(input_image, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            # Mark face as unknown
            results.append((top, right, bottom, left, "Unknown"))
            # Draw a box around the face with a red rectangle
            cv2.rectangle(input_image, (left, top), (right, bottom), (0, 0, 255), 2)
            # Draw label "Unknown"
            label = "Unknown"
            cv2.putText(input_image, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    return input_image, results

def generate_attendance_csv(results, student_encodings, output_csv_path):
    # Initialize attendance dictionary
    attendance = {roll_number: 'A' for roll_number in student_encodings.keys()}
    
    # Mark students present in the results
    for _, _, _, _, roll_number in results:
        if roll_number != "Unknown":
            attendance[roll_number] = 'P'
    
    # Create a DataFrame from the attendance dictionary
    attendance_df = pd.DataFrame(list(attendance.items()), columns=['Roll Number', 'Attendance'])
    
    # Save the DataFrame to a CSV file
    attendance_df.to_csv(output_csv_path, index=False)
    print(f"Attendance has been saved to {output_csv_path}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_images():
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files part'}), 400

    files = request.files.getlist('files[]')
    if not files:
        return jsonify({'error': 'No selected files'}), 400

    all_results = []
    processed_images = []
    for file in files:
        if file and file.filename != '':
            # Save the uploaded file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Process the image
            processed_image, results = recognize_faces(file_path)
            all_results.extend(results)

            # Save the processed image
            processed_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_' + file.filename)
            cv2.imwrite(processed_image_path, processed_image)
            processed_images.append('processed_' + file.filename)

    # Consolidate results to mark attendance
    output_csv_path = os.path.join(app.config['UPLOAD_FOLDER'], 'attendance.csv')
    generate_attendance_csv(all_results, student_encodings, output_csv_path)

    return jsonify({
        'message': 'Images processed',
        'processed_images': processed_images,  # List of processed images
        'csv': 'attendance.csv'  # Relative path to attendance file
    })

@app.route('/download_csv')
def download_csv():
    csv_path = os.path.join(app.config['UPLOAD_FOLDER'], 'attendance.csv')
    if os.path.exists(csv_path):
        return send_file(csv_path, as_attachment=True)
    else:
        return jsonify({'error': 'CSV file not found'}), 404

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

if __name__ == "__main__":
    app.run(debug=True)

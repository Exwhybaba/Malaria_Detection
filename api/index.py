from flask import Flask, request, render_template, send_file, jsonify
import cv2
from ultralytics import YOLO
import os
import uuid
import zipfile

app = Flask(__name__, template_folder='../templates', static_folder='../static')


# Load the YOLO model globally
model_path = 'best.pt'  # Path to your YOLO model
model = YOLO(model_path)

# Define the upload and output folders
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ZIP_FOLDER = 'zips'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(ZIP_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['ZIP_FOLDER'] = ZIP_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

def allowed_file(filename):
    """
    Check if a file has an allowed extension.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    """
    Homepage with project description and navigation buttons.
    """
    return render_template('home.html')

@app.route('/upload')
def upload_page():
    """
    File upload page.
    """
    return render_template('upload.html')

@app.route('/about')
def about():
    """
    About the team page.
    """
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint for object detection. Accepts one or more image files, processes them, 
    and returns a ZIP file containing all processed images.
    """
    if 'images' not in request.files:
        return jsonify({"error": "No image files provided"}), 400

    files = request.files.getlist('images')  # Get multiple files
    if not files:
        return jsonify({"error": "No selected image files"}), 400

    output_files = []  # To store paths of processed images

    for file in files:
        if file.filename == '':
            continue  # Skip empty files
        if not allowed_file(file.filename):
            continue  # Skip unsupported files (e.g., desktop.ini)

        # Save the uploaded image to a temporary location
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Ensure directory exists
        file.save(file_path)

        # Load the image
        image = cv2.imread(file_path)
        if image is None:
            os.remove(file_path)
            continue

        # Perform predictions
        results = model(file_path)

        # Extract results
        boxes = results[0].boxes.xyxy.tolist()  # Bounding boxes
        classes = results[0].boxes.cls.tolist()  # Class indices
        confidences = results[0].boxes.conf.tolist()  # Confidence scores
        names = results[0].names  # Class names dictionary

        # Convert the image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Draw bounding boxes and labels on the image
        for box, cls, conf in zip(boxes, classes, confidences):
            x1, y1, x2, y2 = map(int, box)
            label = f"{names[int(cls)]} {conf:.2f}"

            # Draw bounding box
            cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Add label above the box
            cv2.putText(image_rgb, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Convert RGB back to BGR for saving
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Save the processed image
        output_file_path = os.path.join(app.config['OUTPUT_FOLDER'], f"processed_{uuid.uuid4().hex}.jpg")
        cv2.imwrite(output_file_path, image_bgr)
        output_files.append(output_file_path)

        # Delete the uploaded file after processing
        os.remove(file_path)

    if output_files:
        # Create a ZIP file for the processed images
        zip_filename = os.path.join(app.config['ZIP_FOLDER'], f"processed_images_{uuid.uuid4().hex}.zip")
        with zipfile.ZipFile(zip_filename, 'w') as zipf:
            for file_path in output_files:
                zipf.write(file_path, os.path.basename(file_path))
                os.remove(file_path)  # Remove the processed file after zipping

        return send_file(zip_filename, mimetype='application/zip', as_attachment=True, download_name="processed_images.zip")

    return jsonify({"error": "No valid images processed"}), 500

# Ensure the app callable is exposed for platforms like Vercel
app=app

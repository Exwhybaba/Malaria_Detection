from flask import Flask, request, render_template, send_file, jsonify, url_for
import cv2
from ultralytics import YOLO
import os
import uuid
import zipfile
from tqdm import tqdm
import pandas as pd
import threading
import time
from collections import Counter
from flask_cors import CORS

app = Flask(__name__, template_folder='../templates', static_folder='../static')
CORS(app)  # This allows all domains to access your Flask app

# Load the YOLO model globally
model_path = 'best.pt'  # Path to your YOLO model
model = YOLO(model_path)

# Define the upload and output folders
BASE_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
UPLOAD_FOLDER = os.path.join(BASE_FOLDER, 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_FOLDER, 'outputs')
ZIP_FOLDER = os.path.join(BASE_FOLDER, 'static', 'zips')  # Save ZIP in static folder
CSV_FOLDER = os.path.join(BASE_FOLDER, 'static', 'csv')  # Save CSV in static folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(ZIP_FOLDER, exist_ok=True)
os.makedirs(CSV_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['ZIP_FOLDER'] = ZIP_FOLDER
app.config['CSV_FOLDER'] = CSV_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

def allowed_file(filename):
    """
    Check if a file has an allowed extension.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def schedule_deletion(file_path, delay=120):
    """
    Schedule deletion of a file after a delay (in seconds).
    """
    def delete_file():
        time.sleep(delay)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"File {file_path} deleted after {delay} seconds.")

    thread = threading.Thread(target=delete_file)
    thread.daemon = True  # Ensure the thread doesn't block the app from exiting
    thread.start()

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
    and returns a ZIP file containing all processed images, along with a summary table.
    """
    if 'images' not in request.files:
        return jsonify({"error": "No image files provided"}), 400

    files = request.files.getlist('images')  # Get multiple files
    if not files:
        return jsonify({"error": "No selected image files"}), 400

    output_files = []  # To store paths of processed images
    table_data = []
    total_count = 0
    success_count = 0
    fail_count = 0

    for file in tqdm(files, desc="Processing images"):
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
            fail_count += 1
            continue

        # Perform predictions
        results = model(file_path)

        # Extract results
        boxes = results[0].boxes.xyxy.tolist()  # Bounding boxes
        classes = results[0].boxes.cls.tolist()  # Class indices
        confidences = results[0].boxes.conf.tolist()  # Confidence scores
        names = results[0].names  # Class names dictionary

        # Count occurrences of each class detected in the image
        class_counter = Counter()
        for cls in classes:
            class_name = names[int(cls)]
            class_counter[class_name] += 1

        # Draw bounding boxes and labels on the image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for box, cls, conf in zip(boxes, classes, confidences):
            x1, y1, x2, y2 = map(int, box)
            label = f"{names[int(cls)]} {conf:.2f}"

            # Draw bounding box
            cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (255, 0, 0), 4)

            # Add label above the box
            cv2.putText(image_rgb, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        # Convert RGB back to BGR for saving
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Save the processed image
        output_file_path = os.path.join(app.config['OUTPUT_FOLDER'], f"processed_{uuid.uuid4().hex}.jpg")
        cv2.imwrite(output_file_path, image_bgr)
        output_files.append(output_file_path)

        # Extract the image ID from the file path 
        image_id = os.path.basename(file.filename).split('.')[0]

        # Get the count for WBC and Trophozoite (default to 0 if not found)
        wbc_count = class_counter.get('WBC', 0)
        trophozoite_count = class_counter.get('Trophozoite', 0)

        # Add the data to the table
        table_data.append({
            'image_id': image_id,
            'WBC': wbc_count,
            'Trophozoite': trophozoite_count
        })

        total_count += 1
        success_count += 1 if wbc_count > 0 or trophozoite_count > 0 else 0

        # Delete the uploaded file after processing
        os.remove(file_path)

    if output_files:
        # Create a ZIP file for processed images in the static folder
        zip_filename = os.path.join(app.config['ZIP_FOLDER'], f"processed_images_{uuid.uuid4().hex}.zip")
        with zipfile.ZipFile(zip_filename, 'w') as zipf:
            for file_path in output_files:
                zipf.write(file_path, os.path.basename(file_path))
                os.remove(file_path)  # Remove the processed file after zipping

        # Schedule the deletion of the ZIP file
        schedule_deletion(zip_filename, delay=120)

        # Create a pandas DataFrame for the table
        df = pd.DataFrame(table_data)

        # If only one file is uploaded, render the result page with table
        if len(files) == 1:
            table_html = df.to_html(classes='table table-striped', index=False)
            return render_template(
                'result.html',
                table_html=table_html
            )
        else:
            # Save the DataFrame as a CSV file for multiple files
            csv_path = os.path.join(app.config['CSV_FOLDER'], f"summary_{uuid.uuid4().hex}.csv")
            df.to_csv(csv_path, index=False)

            # Schedule the deletion of the CSV file
            schedule_deletion(csv_path, delay=120)

            return render_template(
                'download.html',
                zip_file=url_for('static', filename=f'zips/{os.path.basename(zip_filename)}', _external=True),
                summary_csv=url_for('static', filename=f'csv/{os.path.basename(csv_path)}', _external=True),
                total_count=total_count,
                success_count=success_count,
                fail_count=fail_count
            )

    return jsonify({"error": "No valid images processed"}), 500

@app.route('/download', methods=['GET'])
def download_file():
    file_path = request.args.get('file_path')
    if not file_path or not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404
    response = send_file(file_path, as_attachment=True)
    response.call_on_close(lambda: os.remove(file_path) if os.path.exists(file_path) else None)
    return response

if __name__ == '__main__':
    app.run(debug=True, port=5000)

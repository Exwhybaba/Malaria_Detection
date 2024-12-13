import cv2
from tqdm import tqdm
from ultralytics import YOLO
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import os

def predict(image_input, model_path='best.pt'):
    """
    Detects bounding boxes on single or multiple images, plots the processed image(s),
    and returns the classes detected and the number of times each was detected, 
    along with a table of image_id, WBC, and Trophozoite counts.

    Parameters:
        image_input (str or list of str): Single image path or a list of image paths.
        model_path (str): Path to the YOLO model file.

    Returns:
        tuple: 
            - A list of tuples with class names and counts for each image.
            - A pandas DataFrame with columns `image_id`, `WBC`, `Trophozoite`.
    """
    # Load the YOLO model
    model = YOLO(model_path)

    # Ensure input is a list for unified processing
    if isinstance(image_input, str):
        image_paths = [image_input]
    elif isinstance(image_input, list):
        image_paths = image_input
    else:
        raise ValueError("image_input must be a string (image path) or a list of strings.")

    
    all_class_counts = []
    table_data = []

    for img_path in tqdm(image_paths, desc="Processing images"):
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = model(img_path)

        # Extract results
        boxes = results[0].boxes.xyxy.tolist()  
        classes = results[0].boxes.cls.tolist()  
        confidences = results[0].boxes.conf.tolist()  
        names = results[0].names  

        # Count occurrences of each class detected in the image
        class_counter = Counter()
        for cls in classes:
            class_name = names[int(cls)]
            class_counter[class_name] += 1

        # Draw detections on the image (making bounding box thicker and bold)
        for box, cls, conf in zip(boxes, classes, confidences):
            x1, y1, x2, y2 = map(int, box)
            label = f"{names[int(cls)]} {conf:.2f}"

            
            cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (255, 0, 0), 4)  

            # Add bold label above the box (using a larger font size and bold)
            cv2.putText(image_rgb, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)  

        # Extract the image ID from the file path 
        image_id = os.path.basename(img_path).split('.')[0].replace("id_", "")

        # Get the count for WBC and Trophozoite (default to 0 if not found)
        wbc_count = class_counter.get('WBC', 0)
        trophozoite_count = class_counter.get('Trophozoite', 0)

        # Add the data to the table
        table_data.append({
            'image_id': image_id,
            'WBC': wbc_count,
            'Trophozoite': trophozoite_count
        })

        # Format class counts to display on the image
        class_labels = [f"{class_name} detected in {count} times" for class_name, count in class_counter.items()]
        class_label_text = '\n'.join(class_labels) if class_labels else "No objects detected"

        
        plt.figure(figsize=(15, 15))  
        plt.imshow(image_rgb)
        plt.title(f"Detected Classes for {image_id}:\n{class_label_text}")
        plt.axis("off")  
        plt.show()
 

        # Store the class counts for this image
        all_class_counts.append(class_counter)

    # Create a pandas DataFrame for the table
    df = pd.DataFrame(table_data)

    # Return the class counts for each image and the table
    return all_class_counts, df
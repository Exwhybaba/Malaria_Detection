import cv2
from tqdm import tqdm
from ultralytics import YOLO
import matplotlib.pyplot as plt


def img_predict(image_input, model_path='best.pt'):
    """
    Detects bounding boxes on single or multiple images and plots the processed image(s).

    Parameters:
        image_input (str or list of str): Single image path or a list of image paths.
        model_path (str): Path to the YOLO model file.

    Returns:
        None: The function plots the image(s) with bounding boxes.
    """
    # Load the YOLO model
    model = YOLO(model_path)

    if isinstance(image_input, str):
        image_paths = [image_input]
    elif isinstance(image_input, list):
        image_paths = image_input
    else:
        raise ValueError("image_input must be a string (image path) or a list of strings.")

    for img_path in tqdm(image_paths, desc="Processing images"):
        # Load the image
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Convert image to RGB for plotting
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Perform predictions
        results = model(img_path)

        # Extract results
        boxes = results[0].boxes.xyxy.tolist()  # Bounding boxes in xyxy format
        classes = results[0].boxes.cls.tolist()  # Class indices
        confidences = results[0].boxes.conf.tolist()  # Confidence scores
        names = results[0].names  # Class names dictionary

        # Draw detections on the image
        for box, cls, conf in zip(boxes, classes, confidences):
            x1, y1, x2, y2 = map(int, box)
            label = f"{names[int(cls)]} {conf:.2f}"

            # Draw bounding box
            cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Add label above the box
            cv2.putText(image_rgb, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Plot the image
        plt.figure(figsize=(10, 10))
        plt.imshow(image_rgb)
        plt.axis("off")
        plt.title(f"Detections for {img_path}")
        plt.show()

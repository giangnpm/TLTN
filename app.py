from flask import Flask, request, jsonify, render_template, send_from_directory
from ultralytics import YOLO
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import cv2
import torch
import numpy as np
import os

app = Flask(__name__)

# Load YOLOv8 model
model_path = "models/best.pt"  # Update with your model path
model = YOLO(model_path)

# Load ViT model and processor
vit_model = ViTForImageClassification.from_pretrained("models/vit-base-oxford-iiit-pets")
vit_processor = ViTImageProcessor.from_pretrained("models/vit-base-oxford-iiit-pets")

# Temporary folder for saving processed images
TEMP_FOLDER = "temp_images"
os.makedirs(TEMP_FOLDER, exist_ok=True)

# Product dictionary
PRODUCTS = {
    "01": "PEPSI Lon 320ml",
    "02": "PEPSI Lốc 6 Lon 320ml",
    "03": "PEPSI Chai nhựa 390ml",
    "05": "PEPSI Không Calo Lon 320ml",
    "07": "PEPSI Không Calo Chai nhựa 390ml",
    "09": "PEPSI Vị Chanh Không Calo Lon 320ml",
    "11": "PEPSI Vị Chanh Không Calo Chai nhựa 390ml",
    "13": "7UP Lon 320ml",
    "14": "7UP Lốc 6 Lon 320ml",
    "15": "7UP Chai nhựa 390ml",
    "17": "MIRINDA Cam Lon 320ml",
    "19": "MIRINDA Cam Chai 390ml",
    "21": "MIRINDA Xá Xị Lon 320ml",
    "22": "MIRINDA Xá Xị Lốc 6 Lon 320ml",
    "23": "MIRINDA Xá Xị Chai 390ml",
    "25": "MIRINDA Soda Kem Lon 320ml",
    "26": "MIRINDA Soda Kem Lốc 6 Lon 320ml",
    "27": "MIRINDA Soda Kem Chai 390ml",
    "29": "STING Dâu Lon 320ml",
    "31": "STING Dâu Chai 330ml",
    "33": "STING Vàng Lon 320ml",
    "35": "STING Vàng Chai 330ml",
    "37": "Ô LONG TEA PLUS Chai 320ml",
    "39": "Ô LONG TEA PLUS Chanh Chai 450ml",
    "41": "AQUAFINA Lon Soda 320ml",
    "43": "REVIVE Chai 500ml",
    "45": "REVIVE Chanh Muối Chai 390ml",
    "47": "TWISTER Chai 445ml",
    "48": "TWISTER Lon 320ml"
}
@app.route('/process-image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files['image']
    image_path = os.path.join(TEMP_FOLDER, file.filename)
    file.save(image_path)

    # Load the image
    orig_image = cv2.imread(image_path)

    # Perform YOLOv8 inference
    results = model(image_path)

    table_data = []
    product_counts = {}

    # Iterate over detected objects
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
        scores = result.boxes.conf.cpu().numpy()  # Confidence scores
        classes = result.boxes.cls.cpu().numpy()  # Class IDs

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)  # Convert to integers
            cropped_image = orig_image[y1:y2, x1:x2]  # Crop the detected object

            # Convert cropped image to PIL format for ViT
            pil_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

            # Process image for ViT
            inputs = vit_processor(images=pil_image, return_tensors="pt")
            outputs = vit_model(**inputs)
            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()
            predicted_label = vit_model.config.id2label[predicted_class_idx]  # Get label directly from ViT

            # Draw bounding box and new label on the image
            cv2.rectangle(orig_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
            cv2.putText(orig_image, predicted_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Add to table data
            product_name = PRODUCTS.get(predicted_label, "Unknown Product")
            # Count the product
            if product_name != "Unknown Product":
                if product_name in product_counts:
                    product_counts[product_name] ["count"]+= 1
                else:
                    product_counts[product_name] = {"count": 1, "label": predicted_label}

            count = product_counts.get(product_name, 0)

            table_data = []
            for name, data in product_counts.items():
                table_data.append({
                    "label": data["label"],
                    "name": name,
                    "count": data["count"],
                    "coordinates": (x1, y1)
                })

    # Sort table_data by position (top to bottom, left to right)
    # table_data.sort(key=lambda item: (item["coordinates"][1], item["coordinates"][0]))
    table_data.sort(key=lambda item: (round(item["coordinates"][1] / 10), item["coordinates"][0]))

    # Save the processed image
    processed_image_path = os.path.join(TEMP_FOLDER, f"processed_{file.filename}")
    cv2.imwrite(processed_image_path, orig_image)

    return render_template('index.html', uploaded_image=file.filename, processed_image=f"processed_{file.filename}", table_data=table_data, product_counts=product_counts)

@app.route('/temp_images/<filename>')
def temp_images(filename):
    return send_from_directory(TEMP_FOLDER, filename)

@app.route('/')
def index():
    return render_template('index.html', uploaded_image=None, processed_image=None, table_data=[])

if __name__ == '__main__':
    app.run(debug=True)

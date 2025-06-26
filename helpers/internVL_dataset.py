import json
import os
from PIL import Image  # Added to get image dimensions

# Input and output file paths
dataset_path = "../../datasets/mmsd2"
input_file = os.path.join(dataset_path, "valid.json")
output_file = os.path.join(dataset_path, "valid.jsonl")
image_dir = os.path.join(dataset_path, "dataset_image/")

# Load the input JSON
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Write to JSONL
with open(output_file, "w", encoding="utf-8") as f_out:
    for idx, item in enumerate(data):
        image_id = item["image_id"]
        text = item["text"]
        label = int(item["label"])
        instruction = f"Classify the text <{text}> and the image into one of the following categories: <SARCASTIC, NOT SARCASTIC>."
        label_text = "SARCASTIC" if label == 1 else "NOT SARCASTIC"
        image_path = os.path.join(image_dir, f"{image_id}.jpg")

        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except FileNotFoundError:
            print(f"Warning: Image not found for ID {image_id}")
            width, height = 0, 0  # Fallback values

        jsonl_item = {
            "id": idx,
            "image": f"{image_id}.jpg",
            "width": width,
            "height": height,
            "conversations": [
                {"from": "human", "value": f"<image>\n{instruction}"},
                {"from": "gpt", "value": f"{label_text}"}
            ]
        }

        f_out.write(json.dumps(jsonl_item) + "\n")

import json
import os
from PIL import Image  # Added to get image dimensions

# Input and output file paths
dataset_path = "../../datasets/mmsd2"
input_file = os.path.join(dataset_path, "valid.json")
output_file = os.path.join(dataset_path, "pixtral_finetuning_format/valid.json")
image_dir = os.path.join(dataset_path, "dataset_image/")

# Load the input JSON
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

with open(output_file, "w", encoding="utf-8") as f_out:
    result = []
    for idx, item in enumerate(data):
        image_id = item["image_id"]
        text = item["text"]
        label = int(item["label"])
        instruction = f"Classify the text <{text}> and the image into one of the following categories: <SARCASTIC, NOT SARCASTIC>."
        label_text = "SARCASTIC" if label == 1 else "NOT SARCASTIC"
        image_path = os.path.join(image_dir, f"{image_id}.jpg")

        json_item = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image_path": os.path.abspath(image_path)},
                        {"type": "text", "text": instruction}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": label_text}
                    ]
                }
            ]
        }
        result.append(json_item)
    f_out.write(json.dumps(result, indent = 2))

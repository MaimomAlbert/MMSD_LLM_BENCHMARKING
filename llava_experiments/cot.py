# Chain of Thought prompting with Llama
# Llava -> Structured output
import torch
from PIL import Image
from transformers import pipeline
import pandas as pd
from tqdm import tqdm
import json
import os
import re
from prompts import ( 
    cot_text_instruction, 
    cot_image_instruction, 
    cot_mm_instruction,
    non_cot_text_instruction,
    non_cot_image_instruction,
    non_cot_mm_instruction
)
import logging

os.environ['CURL_CA_BUNDLE'] = ''
# Configure the logging system
logging.basicConfig(
    filename='/home/gpuuser3/sinngam_albert/work/llava_experiments/logs/cot.log',
    filemode='a',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Huggingface Model Id
MODEL_ID = "llava-hf/llava-v1.6-vicuna-13b-hf"
pipe = pipeline("image-text-to-text", model=MODEL_ID)

def get_llava_next_response(
        text: str,
        image_path: str,
        instruction:str
        ):
    """
        text(str): Utterance accompanying the image
        image_path(str): URL of the image
        instruction(str): User instruction to the model
    """
    # Create input messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "text", "text": text},
                {"type": "image", "url": image_path}
            ]
        },
    ]
    if image_path == None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "text", "text": text}
                ]
            },
        ]
    elif text == None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image", "url": image_path}
                ]
            },
        ]
    out = pipe(text=messages, max_new_tokens=1000)
    return out[0]['generated_text'][-1]["content"]

# Helper functions
def convert_label_to_text(label):
    if label == 0:
        return "NOT SARCASTIC"
    elif label == 1:
        return "SARCASTIC"
    else:
        return "UNKNOWN"

def convert_text_to_label(text):
    if text == "NOT SARCASTIC":
        return 0
    elif text == "SARCASTIC":
        return 1
    else:
        return -1
    
def append_to_json_output(
        sample, 
        label,
        reasoning,
        modality,
        output_file
    ):
    """
    sample: dataframe row
    label: 0 / 1
    reasoning: str
    modality: text/image/mm
    output_file: where to save
    """
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            try:
                cur_samples = json.load(f)
            except json.JSONDecodeError:
                cur_samples = []
    else:
        cur_samples = []
    
    if modality == "text":
        target_label = sample["text_label"]
    elif modality == "image":
        target_label = sample["image_label"]
    elif modality == "mm":
        target_label = sample["multimodal_label"]

    new_sample = {
        "image_id": str(sample['image_id']),
        "text": str(sample['text']),
        f"llava-{modality}-label": str(label),
        f"llava-{modality}-reasoning": str(reasoning),
        f"target-{modality}-label": str(target_label),
    }

    cur_samples.append(new_sample)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(cur_samples, f, ensure_ascii=False, indent=4)

def get_last_index(file_path="track.txt"):
    if not os.path.exists(file_path):
        return 0
    
    with open(file_path, "r", encoding="utf-8") as f:
        index = f.read().strip()

    return int(index) if index.isdigit() else 0
def update_last_index(index, file_path="track.txt"):
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(str(index))

def extract_label_and_reasoning(text):
    # Look for Label with or without colon, with flexible whitespace
    label_pattern = r"\*\*Label\*\*\s*:?\s*(.+?)(?:\n|$)"
    label_match = re.search(label_pattern, text)
    
    # Look for Reasoning with or without colon, capturing all content until next section or end
    reasoning_pattern = r"\*\*Reasoning\*\*\s*:?\s*([\s\S]+?)(?:\n\n\*\*|\Z)"
    reasoning_match = re.search(reasoning_pattern, text)
    
    # Set default values
    label = "-1"
    reasoning = "undetermined"
    
    # Update with matched values if found
    if label_match:
        label = label_match.group(1).strip()
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
    
    return label, reasoning


if __name__ == "__main__":
    ###################################################
    # Edit here!
    img_dir = "/home/gpuuser3/sinngam_albert/datasets/Model-Prompt-Evaluation-Dataset/triplets_images"
    track_file = "/home/gpuuser3/sinngam_albert/work/llava_experiments/track.txt"
    output_file = "/home/gpuuser3/sinngam_albert/work/llava_experiments/generated_annotations/triplets_image_non_cot.json"
    dataset = "/home/gpuuser3/sinngam_albert/datasets/Model-Prompt-Evaluation-Dataset/triplets.csv"
    image_extension = "png"
    modality = "image"
    prompt = non_cot_image_instruction
    ###################################################
    df = pd.read_csv(dataset, dtype=str)

    start = get_last_index(file_path=track_file)
    for i in range(start, len(df)):
        try:
            sample = df.loc[i]
            if os.path.exists(f"{img_dir}/{sample['image_id']}.{image_extension}"):
                model_response = get_llava_next_response(
                                    text = None,
                                    image_path = f"{img_dir}/{sample['image_id']}.{image_extension}",
                                    instruction=prompt
                                )

                ## Logging
                print(f"### Index: {i}")
                print(f"### Submission ID: {sample['image_id']}")
                print(f"### Text: {sample['text']}")
                print(f"### Model Response: {model_response}")
               
                label, reasoning = extract_label_and_reasoning(model_response)
                print(f"### Extracted Label: {label}")
                print(f"### Extracted Reasoning: {reasoning}")
                append_to_json_output(
                    sample = sample,
                    label=convert_text_to_label(label),
                    reasoning = reasoning,
                    modality = modality,
                    output_file=output_file
                )
                print(f"### Saved to {output_file}")
                update_last_index(i, file_path=track_file)
                print(f"### Updated next index to {i+1}")
                print("-------------------------------------------")
        except Exception as e:
            print.info(f"An error occurred at index {i}: {e}")
            print(traceback.print_exc())

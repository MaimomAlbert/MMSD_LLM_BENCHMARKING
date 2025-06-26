# Chain of Thought prompting with Llama
# Llama -> Structured output
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
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
import traceback


os.environ['CURL_CA_BUNDLE'] = ''

# Configure the logging system
logging.basicConfig(
    filename='/home/gpuuser3/sinngam_albert/work/llama_experiments/logs/cot_2.log',
    filemode='a',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


# Huggingface Model Id
model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_id)


def get_llama_response(
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
    image = None
    if image_path == None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "text", "text": f"Text: {text}"}
                ]
            },
        ]
    elif text == None:
        image = Image.open(image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image"}
                ]
            },
        ]
    else:
        image = Image.open(image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "text", "text": text},
                    {"type": "image"}
                ]
            },
        ]

    # Prepare inputs
    input_text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True
    )
    inputs = processor(image, input_text, add_special_tokens=False,return_tensors="pt").to(model.device)

    # Generate outputs
    output = model.generate(**inputs, max_new_tokens=1000)
    return processor.decode(output[0])

def parse_eot_content(text):
    pattern = r"<\|start_header_id\|>assistant<\|end_header_id\|>\s*(.*?)\s*<\|eot_id\|>"
    
    try:
        # Use re.DOTALL to match across multiple lines
        match = re.search(pattern, text, re.DOTALL)
        
        if match:
            return match.group(1).strip()
        else:
            return "Assistant output not found."
    except Exception as e:
        return f"Error extracting assistant output: {str(e)}"


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
        f"llama-{modality}-label": str(label),
        f"llama-{modality}-reasoning": str(reasoning),
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
    track_file = "/home/gpuuser3/sinngam_albert/work/llama_experiments/track.txt"
    output_file = f"/home/gpuuser3/sinngam_albert/work/llama_experiments/generated_annotations/triplets_image_cot.json"
    dataset = "/home/gpuuser3/sinngam_albert/datasets/Model-Prompt-Evaluation-Dataset/triplets.csv"
    image_extension = "png"
    modality = "image"
    prompt = cot_image_instruction
    ###################################################
    df = pd.read_csv(dataset, dtype=str)

    start = get_last_index(file_path=track_file)
    for i in range(start, len(df)):
        try:
            sample = df.loc[i]
            if os.path.exists(f"{img_dir}/{sample['image_id']}.{image_extension}"):
                model_response = get_llama_response(
                                    text = None,
                                    image_path = f"{img_dir}/{sample['image_id']}.{image_extension}",
                                    instruction=prompt
                                )
                model_response = parse_eot_content(model_response)

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

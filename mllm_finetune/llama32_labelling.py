import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
import pandas as pd
from tqdm import tqdm
import json
import os
import re
import traceback


model_id = "/home/gpuuser3/sinngam_albert/work/mllm_finetune/llama32_11b_mmsd2_outputs_merged"
model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_id)


def get_llama_response(
        text: str,
        image_path: str
        ):
    """
        text(str): Utterance accompanying the image
        image_path(str): URL of the image
    """
    image = Image.open(image_path)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text},
            ]
        },
    ]
    input_text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True
    )
    inputs = processor(image, input_text, add_special_tokens=False,return_tensors="pt").to(model.device)
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
    
def append_to_json_output(image_id: str, text: str, subreddit: str, gpt: str, output_file: str):
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            try:
                cur_samples = json.load(f)
            except json.JSONDecodeError:
                cur_samples = []
    else:
        cur_samples = []
    new_sample = {
        "image_id": image_id,
        "text": text,
        "subreddit": subreddit,
        "gpt": gpt
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

def convert_text_to_label(response):
    if response == "Sarcastic":
        return 1
    elif response == "Non-sarcastic":
        return 0
    else:
        return -1

if __name__ == "__main__":
    df = pd.read_csv("/home/gpuuser3/sinngam_albert/datasets/osint/batch_2.csv", dtype='str')
    df['image_path'] = df['image_id'].apply(
        lambda x: f"/home/gpuuser3/sinngam_albert/datasets/osint/batch_2/{x}.png"
    )
    df['question'] = df['text'].apply(
        lambda x: f"Classify the text <{x}> and the image into one of: <Sarcastic, Non-sarcastic>."
    )
    track_file = "track.txt"
    start = get_last_index(file_path=track_file)
    print(start)
    print(len(df))
    for i in range(start, len(df)):
        try:
            sample = df.loc[i]
            if os.path.exists(sample['image_path']):
                model_response = parse_eot_content(get_llama_response(
                                    text = sample['question'],
                                    image_path = sample['image_path']
                                ))
                print(f"Processing row {i+1}/{len(df)}")
                print(f"Question: {sample['question']}\nResponse: {model_response}\n")
               
                append_to_json_output(
                    image_id=sample['image_id'],
                    text=sample['text'],
                    subreddit=sample['subreddit'],
                    gpt= str(convert_text_to_label(model_response)),
                    output_file="gpt_predictions/llama32_11B_osint_batch_2.json"
                )
                update_last_index(i, file_path=track_file)
                print("-------------------------------------------")
        except Exception as e:
            print(f"An error occurred at index {i}: {e}")
            print(traceback.print_exc())

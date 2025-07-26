import torch
from PIL import Image
from transformers import pipeline
import pandas as pd
from tqdm import tqdm
import json
import os
import re
import traceback


pipe = pipeline("image-text-to-text", model="/home/gpuuser3/sinngam_albert/work/mllm_finetune/llava16_mistral_7B_mmsd2_outputs_merged")

def clean_output(text):
    if 'not' in text.lower():
        return 'NOT SARCASTIC'
    elif 'sarcastic' in text.lower():
        return 'SARCASTIC'
    elif 'not' not in text.lower() and 'sarcastic' not in text.lower():
        return 'UNKNOWN'

def get_llava_response(
        text: str,
        image_path: str
        ):
    """
        text(str): Utterance accompanying the image
        image_path(str): URL of the image
    """
    messages = [
        {
        "role": "user",
        "content": [
            {"type": "image", "url": image_path},
            {"type": "text", "text": text},
            ],
        },
    ]
    out = pipe(text=messages, max_new_tokens=10)
    result = out[0]['generated_text'][1]['content']
    print(result)

    result = clean_output(result)    
    return result
    
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
    if response == "SARCASTIC":
        return 1
    elif response == "NOT SARCASTIC":
        return 0
    else:
        return -1

if __name__ == "__main__":
    df = pd.read_csv("/home/gpuuser3/sinngam_albert/datasets/osint/batch_2.csv", dtype='str')
    df['image_path'] = df['image_id'].apply(
        lambda x: f"/home/gpuuser3/sinngam_albert/datasets/osint/batch2/{x}.png"
    )
    df['question'] = df['text'].apply(
        lambda x: f"Classify the text <{x}> and the image into one of the following categories: <SARCASTIC, NOT SARCASTIC>."
    )
    track_file = "/Users/sinngamkhaidem/Developer/mllm-based-mmsd-osint/mllm_finetune/track_mistral.txt"
    start = get_last_index(file_path=track_file)
    print(start)
    print(len(df))
    for i in range(start, len(df)):
        try:
            sample = df.loc[i]
            if os.path.exists(sample['image_path']):
                model_response = clean_output(get_llava_response(
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
                    output_file="gpt_predictions/llava16_mistral_osint_batch_2.json"
                )
                update_last_index(i, file_path=track_file)
                print("-------------------------------------------")
        except Exception as e:
            print(f"An error occurred at index {i}: {e}")
            print(traceback.print_exc())

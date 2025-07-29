import torch
from PIL import Image
import torch
from peft import LoraConfig, get_peft_model
from transformers import IdeficsForVisionText2Text, AutoProcessor, BitsAndBytesConfig
import torchvision.transforms as transforms
import pandas as pd
from tqdm import tqdm
import json
import os
import re
import traceback


device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint = "/home/gpuuser3/sinngam_albert/work/mllm_finetune/idefics_9B_MMSD_outputs_merged"
processor = AutoProcessor.from_pretrained(checkpoint)
model = IdeficsForVisionText2Text.from_pretrained(checkpoint, device_map="auto")


def get_idefics_response(
        text: str,
        image_path: str
        ):
    """
        text(str): Utterance accompanying the image
        image_path(str): URL of the image
    """
    image = Image.open(image_path)
    prompts = [
        image,
        text
    ]
    tokenizer = processor.tokenizer
    bad_words = ["<image>", "<fake_token_around_image>"]
    if len(bad_words) > 0:
        bad_words_ids = tokenizer(bad_words, add_special_tokens=False).input_ids

    eos_token = "</s>"
    eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)

    inputs = processor(prompts, return_tensors="pt").to(device)
    generated_ids = model.generate(**inputs, eos_token_id=[eos_token_id], bad_words_ids=bad_words_ids, max_new_tokens=15, early_stopping=True)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    answer = generated_text.split("Answer:")[-1].strip()
    print(answer)
    return answer

def clean_response(text):
    if text in ["SARCASTIC", "NOT SARCASTIC"]:
        return text
    else:
        return "UNKNOWN"

def append_to_json_output(image_id: str, text: str, subreddit: str, gpt: str, output_file: str):
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            cur_samples = json.load(f)
                
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
        lambda x: f"/home/gpuuser3/sinngam_albert/datasets/osint/batch_2/{x}.png"
    )
    df['question'] = df['text'].apply(
        lambda x: f"Question: Classify the text <{x}> and the image into one of the following categories: <SARCASTIC, NOT SARCASTIC>. Answer:"
    )
    track_file = "/home/gpuuser3/sinngam_albert/work/mllm_finetune/track_idefics.txt"
    start = get_last_index(file_path=track_file)
    print(start)
    print(len(df))
    for i in range(start, len(df)):
        try:
            sample = df.loc[i]
            if os.path.exists(sample['image_path']):
                model_response = clean_response(get_idefics_response(
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
                    output_file="gpt_predictions/idefics2_9B_osint_batch_2.json"
                )
                update_last_index(i, file_path=track_file)
                print("-------------------------------------------")
        except Exception as e:
            print(f"An error occurred at index {i}: {e}")
            print(traceback.print_exc())

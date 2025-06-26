import torch
from PIL import Image
from transformers import pipeline
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (accuracy_score, 
                             classification_report, 
                             confusion_matrix)
import numpy as np
import re
import pandas as pd


pipe = pipeline("image-text-to-text", model="/home/gpuuser3/sinngam_albert/work/mllm_finetune/llava15_7B_mmsd2_outputs_merged")

def clean_output(text):
    """
    Clean text into one of 'SARCASTIC' or 'NOT SARCASTIC' by mapping using regex.
    """
    text = text.strip().upper()
    if re.search(r'\bNOT SARCASTIC\b', text):
        return 'NOT SARCASTIC'
    elif re.search(r'\bSARCASTIC\b', text):
        return 'SARCASTIC'
    else:
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
    out = pipe(text=messages, max_new_tokens=16)
    result = out[0]['generated_text'][1]['content']
    print(result)

    result = clean_output(result)    
    return result

mmsd_test = pd.read_json("/home/gpuuser3/sinngam_albert/datasets/mmsd2/test.json")
mmsd_test['image_path'] = mmsd_test['image_id'].apply(
    lambda x: f"/home/gpuuser3/sinngam_albert/datasets/mmsd2/dataset_image/{x}.jpg"
)

mmsd_test['question'] = mmsd_test['text'].apply(
    lambda x: f"Classify the text <{x}> and the image into one of the following categories: <SARCASTIC, NOT SARCASTIC>."
)


for idx, row in mmsd_test.iterrows():
    print(f"Processing row {idx+1}/{len(mmsd_test)}")
    response = get_llava_response(
        text=row['question'],
        image_path=row['image_path']
    )
    print(f"Question: {row['question']}\nResponse: {response}\n")
    mmsd_test.at[idx, 'llava15_response'] = response

mmsd_test['llava15_label'] = mmsd_test['llava15_response'].apply(
    lambda x: "0" if "NOT" in x else "1"
)

mmsd_test.to_json(
    "llava15_7B_mmsd2_test_predictions.json",
    orient="records",
    lines=False
)

# Chain of Thought prompting with Llama
# Llava -> Structured output
import torch
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import pandas as pd
from tqdm import tqdm
import json
import os
import re
import logging
import traceback

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
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-13b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-vicuna-13b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True) 
model.to("cuda:0")

def get_llava_next_response(
        text: str,
        image_path: str,
        ):
    """
        text(str): Utterance accompanying the image
        image_path(str): URL of the image
    """
    # Create input messages
    instruction = f"Classify the text <{text}> and the image into one of <Sarcastic, Non-sarcastic>."
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": instruction}
            ]
        }
    ]
    images = [image_path] if image_path != None else None
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

    inputs = processor(images=images, text=prompt, return_tensors="pt").to("cuda:0")

    # autoregressively complete prompt
    output = model.generate(**inputs, max_new_tokens=1000)
    return processor.decode(output[0], skip_special_tokens=True).split("ASSISTANT: ")[-1].strip()
    
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


if __name__ == "__main__":
    ###################################################
    dataset_folder = "../../datasets/mmsd2/"
    img_dir = os.path.join(dataset_folder, "dataset_image")
    track_file = "track.txt"
    output_file = "generated_annotations/llava_mmsd2_test.json"
    dataset = os.path.join(dataset_folder, "test.json")
    image_extension = "jpg"
    ###################################################
    df = pd.read_json(dataset, dtype=str)

    start = get_last_index(file_path=track_file)
    for i in range(start, len(df)):
        try:
            sample = df.loc[i]
            if os.path.exists(f"{img_dir}/{sample['image_id']}.{image_extension}"):
                model_response = get_llava_next_response(
                                    text = sample['text'],
                                    image_path = f"{img_dir}/{sample['image_id']}.{image_extension}",
                                )

                ## Logging
                print(f"### Index: {i}")
                print(f"### Image ID: {sample['image_id']}")
                print(f"### Text: {sample['text']}")
                print(f"### Model Response: {model_response}")
               
                #append_to_json_output(
                #    sample = sample,
                #    label=convert_text_to_label(label),
                #    reasoning = reasoning,
                #    modality = modality,
                #    output_file=output_file
                #)
                print(f"### Saved to {output_file}")
                update_last_index(i, file_path=track_file)
                print(f"### Updated next index to {i+1}")
                print("-------------------------------------------")
        except Exception as e:
            print.info(f"An error occurred at index {i}: {e}")
            print(traceback.print_exc())

from PIL import Image
import pandas as pd
from tqdm import tqdm
import json
import os
import traceback
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

path = '/home/gpuuser3/sinngam_albert/work/internvl_ft/internvl_chat/work_dirs/internvl_chat_v2_5/internvl2_5_8b_dynamic_res_2nd_finetune_lora_mmsd2_merge'
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)


def get_intern_response(
        text: str,
        image_path: str
        ):
    """
        text(str): Utterance accompanying the image
        image_path(str): URL of the image
    """
    # set the max number of tiles in `max_num`
    pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
    generation_config = dict(max_new_tokens=1024, do_sample=False)
    response = model.chat(tokenizer, pixel_values, text, generation_config)
    return response
    
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
    df = pd.read_csv("/home/gpuuser3/sinngam_albert/datasets/osint/unannotated/no_comment.csv", dtype='str')
    df['image_path'] = df['image_id'].apply(
        lambda x: f"/home/gpuuser3/sinngam_albert/datasets/osint/all_images/{x}.png"
    )
    df['question'] = df['text'].apply(
        lambda x: f"<image>\nClassify the text <{x}> and the image into one of the following categories: <SARCASTIC, NOT SARCASTIC>."
    )
    track_file = "track.txt"
    start = get_last_index(file_path=track_file)
    print(start)
    print(len(df))
    for i in range(start, len(df)):
        try:
            sample = df.loc[i]
            if os.path.exists(sample['image_path']):
                model_response = get_intern_response(
                                    text = sample['question'],
                                    image_path = sample['image_path']
                                )
                print(f"Processing row {i+1}/{len(df)}")
                print(f"Question: {sample['question']}\nResponse: {model_response}\n")
               
                append_to_json_output(
                    image_id=sample['image_id'],
                    text=sample['text'],
                    subreddit=sample['subreddit'],
                    gpt= str(convert_text_to_label(model_response)),
                    output_file="gpt_predictions/internvl25_osint.json"
                )
                update_last_index(i, file_path=track_file)
                print("-------------------------------------------")
        except Exception as e:
            print(f"An error occurred at index {i}: {e}")
            print(traceback.print_exc())

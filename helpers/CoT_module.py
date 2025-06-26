
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [0]))
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import logging
import re
import sys
import json
from PIL import ImageFile
import torch
from tqdm import tqdm
import transformers
import torch
from PIL import Image
import requests
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers.utils import logging


logging.set_verbosity_error()

torch.set_grad_enabled(False)

# init model and tokenizer

processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

#####################################################


# logger = logging.getLogger(__name__)
#ImageFile.LOAD_TRUNCATED_IMAGES = True

torch.set_grad_enabled(False)
torch.set_default_tensor_type(torch.cuda.HalfTensor)

def save_json(file_name, data):
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write(json.dumps(data, indent=4))


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def process_response(response):
    # Remove any system messages or instructions
    if "[/INST]" in response:
        response = response.split("[/INST]")[-1]
    else:
        response = response

    lines = response.strip().split("\n")

    # Initialize variables
    label = ''
    reason = ''
    
    # Iterate through lines to find Label and Reason
    for line in lines:
        line = line.strip()
        if line.startswith('Label:'):
            label = line[len('Label:'):].strip()
        elif line.startswith('Reason:'):
            reason = line[len('Reason:'):].strip()
    
    # Handle cases where the model didn't follow instructions
    if not label or not reason:
        label = ''
        reason = ''
    
    return reason, label
    
########################### Image Prompt ################################################
    
# Define the provide_prompt function
def image_prompt() -> list[dict]:
    prompt = """
Given the image, perform the following tasks in order:
Task 1: Properly analyse the image considering any background information and determine if the image is sarcastic. Print 0 if the image is non-sarcastic or 1 if it is sarcastic.
Task 2: Based on the analysis of Task 1, provide a concise reasoning explaining the label provided in Task 1.
Output Format:
Label: [1/0]
Reason: [<reason>]
Note:
Explanation of output format: 
The output of task 1 is Label, only 1/0 should be printed.
The output of task 2 is Reason, only the concise reasoning should be printed.
Adhere strictly to the order and independence of tasks as described. Provide outputs exactly in the specified format.
"""
    
    messages = [{"role": "system", "content": [{"type": "text", "text": prompt}]}]
    return messages

########################### Text Prompt ####################################################

# Define the provide_prompt function
def text_prompt() -> list[dict]:
    prompt = """
Given the text, perform the following tasks in order:
Task 1: Properly analyse the text considering any background information and determine if the text is sarcastic. Print 0 if the text is non-sarcastic or 1 if it is sarcastic.
Task 2: Based on the analysis of Task 1, provide a concise reasoning explaining the label provided in Task 1.
Output Format:
Label: [1/0]
Reason: [<reason>]
Note:
Explanation of output format: 
The output of task 1 is Label, only 1/0 should be printed.
The output of task 2 is Reason, only the concise reasoning should be printed
Adhere strictly to the order and independence of tasks as described. Provide outputs exactly in the specified format.
"""
    messages = [{"role": "system", "content": [{"type": "text", "text": prompt}]}]
    return messages

######################## Multimodal Prompt ####################################################

def multimodal_prompt() -> list[dict]:
    prompt = """
You will be given a **Supporting Text** and an **Image**.

**Definitions**:
- **Supporting Text**: The text provided separately, not embedded in the image.
- **Caption**: The text embedded within the image.

Please **think step by step** and perform the following tasks:

1. **Extract the Caption** from the image.

2. **Analyze** the **Supporting Text**, the **Caption**, and the **Image** together to determine if the combined image-text pair is sarcastic. Consider any contrasts, contradictions, or ironic elements.

3. **Determine the Label**:
   - If the pair is **sarcastic**, print `Label: 1`.
   - If the pair is **non-sarcastic**, print `Label: 0`.

4. **Provide a Reason**:
   - Write a concise explanation for why you labeled the image-text pair as sarcastic or non-sarcastic.

**Output Format**:
Thinking: [Your step-by-step thinking] 
Supporting Text: [Supporting Text] 
Caption in the Image: [Caption] 
Label: [1/0] 
Reason: [Your reasoning]

**Notes**:
- **Adhere strictly** to the order and independence of tasks as described.
- Provide outputs **exactly** in the specified format.
- Do **not** include any additional commentary or explanation outside the specified format.
- Ensure that your **Thinking** section includes your step-by-step analysis.
- Keep your **Reason** concise and directly related to your analysis.
"""
    messages = [{"role": "system", "content": [{"type": "text", "text": prompt}]}]
    return messages


##########################Chat Template################################

def sendMessageMultimodal(instruction, text=None, image=None):
    # Create the conversation with separate content items for text and image
    usercontent = "supporting_text : " + text
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": usercontent if text else ""},
                {"type": "image"},
            ],
        },
    ]

    conversation = instruction + conversation

    # print(f"Conversation: {conversation}")

    # Generate the prompt
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    # Prepare the inputs
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(device)

    # Autoregressively complete the prompt
    try:
        output = model.generate(**inputs, max_new_tokens=1024)
    except:
        return ''
    response_text = processor.decode(output[0], skip_special_tokens=True)


    return response_text


def sendMessage(instruction, text=None, image_path=None):
    # Prepare the image
    #image = None
    # Create the conversation based on initial history and new message
    conversation = [
         {
            "role": "user",
            "content": [
                {"type": "text", "text": ""},
                {"type": "image"},
            ],
        },
    ]

    if text is not None:
        conversation[0]['content'][0]['text'] = text
        
    conversation = instruction + conversation

   #print(conversation)

    # Generate the prompt
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    # Prepare the inputs
    inputs = processor(prompt, image, return_tensors="pt").to(device)

    # Autoregressively complete the prompt
    try:
        output = model.generate(**inputs, max_new_tokens=200)
    except:
        return ''
    response_text = processor.decode(output[0], skip_special_tokens=True)

    return response_text

#######################################################################


# =======================CoT module=======================

#process_file_list = ['train_data', 'val_data', 'test_data']
process_file_list = ['new_train_data']

data_path = './data/mmsd2'
image_file_path = './data/image_data'
new_data_path = './data/mmsd2'

for file in process_file_list:
    # Check first if there are any checkpoints present
    print(f'File:{file}')
    if os.path.exists(f'{data_path}/{file}.json'):
        #datas = load_json(f'{new_data_path}/new_{file}.json')
        datas = load_json(f'{data_path}/{file}.json')
        # print(f"datas : {datas}")
    else:
        print(f'File paths not found')
    count = 0
    for line in tqdm(datas):
        #print(line)
        if 'llava_mix_info' in line:
            continue
        img_name = line['images_name']
        text = line['text']
        if not os.path.exists(f'{image_file_path}/{img_name}'):
            continue
       
        img = f'{image_file_path}/{img_name}'
        image = Image.open(img)

        instruction = image_prompt()
        response1 = sendMessage(instruction, None, image)
        #print(f'Response1 {response1}')
        reason1, label1 = process_response(response1)
        if reason1 == '' or label1 == '':
            line['llava_img_response'] = response1
        line['llava_img_info'] = reason1
        line['llava_img_label'] = label1

        instruction = text_prompt()
        response2 = sendMessage(instruction, text, None)
        #print(f'Response1 {response2}')
        reason2, label2 = process_response(response2)
        if reason2 == '' or label2 == '':
            line['llava_text_response'] = response2
        line['llava_text_info'] = reason2
        line['llava_text_label'] = label2

        instruction = multimodal_prompt()
        response3 = sendMessageMultimodal(instruction, text, image)
        #print(f'Response1 {response3}')
        reason3, label3 = process_response(response3)
        if reason3 == '' or label3 == '':
            line['llava_mix_response'] = response3
        line['llava_mix_info'] = reason3
        line['llava_mix_label'] = label3


        # print('*****************************')
        # print(f"Image name: {img_name}")
        # print(f"Reason1 {reason1}")
        # print(f"Label1 {label1}")
        # print(f"Reason2 {reason2}")
        # print(f"Label2 {label2}")
        # print(f"Reason3 {reason3}")
        # print(f"Label3 {label3}")
        # print('*****************************')

        count += 1
        if count == 50:
            # This is a checkpoint to prevent unexpected stops during code execution and loss of previous results.
            print('save_a_part')
            count = 0
            save_json(f'{new_data_path}/new_{file}.json', datas)

    save_json(f'{new_data_path}/new_{file}.json', datas)

print('finish!')

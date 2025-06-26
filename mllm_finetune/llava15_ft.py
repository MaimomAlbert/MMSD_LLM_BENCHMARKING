from unsloth import FastVisionModel # FastLanguageModel for LLMs
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
import os
from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv('HUGGINGFACE_TOKEN')
print(f"#### Huggingface token loaded: {hf_token}")

print("#### Creating model and tokenizers.")
model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/llava-1.5-7b-hf",
    load_in_4bit = False, # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
)
tokenizer.num_additional_image_tokens = 1
print("#### Model, Tokenizers created.")

print("#### Getting peft model.")
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # False if not finetuning vision layers
    finetune_language_layers   = True, # False if not finetuning language layers
    finetune_attention_modules = True, # False if not finetuning attention layers
    finetune_mlp_modules       = True, # False if not finetuning MLP layers
    r = 16,           # The larger, the higher the accuracy, but might overfit
    lora_alpha = 16,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
    # target_modules = "all-linear", # Optional now! Can specify a list if needed
)

print("#### Preparing dataset.")
data_folder = "../../datasets/mmsd2/"
train_dataset = os.path.join(data_folder, "train.json")
valid_dataset = os.path.join(data_folder, "valid.json")

train_df = pd.read_json(train_dataset)
valid_df = pd.read_json(valid_dataset)

def create_img_paths(id):
    img_directory = "/home/gpuuser3/sinngam_albert/datasets/mmsd2/dataset_image"
    path = f"{img_directory}/{id}.jpg"
    return path

def convert_label_to_text(label):
    if label == 0:
        return "NOT SARCASTIC"
    else:
        return "SARCASTIC"
    
train_df["image"] = train_df["image_id"].apply(lambda x: create_img_paths(x))
valid_df["image"] = valid_df["image_id"].apply(lambda x: create_img_paths(x))

print("#### Converting dataset to conversation format.")
def convert_to_conversation(sample):
    prompt = f"Classify the text <{sample['text']}> and the image into one of the following categories: <SARCASTIC, NOT SARCASTIC>."
    conversation = [
        { "role": "user",
          "content" : [
            {"type" : "image", "image" : Image.open(sample["image"])},
            {"type": "text", "text": prompt}
          ]
        },
        { "role" : "assistant",
          "content" : [
            {"type" : "text",  "text"  : convert_label_to_text(sample['label'])}
          ]
        },
    ]
    return { "messages" : conversation }

converted_dataset_train = [
    convert_to_conversation(sample) for i, sample in tqdm(train_df.iterrows(), total=len(train_df))
]
converted_dataset_valid = [
    convert_to_conversation(sample) for i, sample in tqdm(valid_df.iterrows(), total=len(valid_df))
]
print("#### Dataset preparation completed.")


print("""#### Training the model""")
FastVisionModel.for_training(model) # Enable for training!
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = UnslothVisionDataCollator(model, tokenizer), # Must use!
    train_dataset = converted_dataset_train,
    args = SFTConfig(
        do_eval = False,
        per_device_train_batch_size = 12,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 30, # Set this for quick testing
        #num_train_epochs = 1, # Set this instead of max_steps for full training runs
        learning_rate = 2e-4,
        fp16 = not is_bf16_supported(),
        bf16 = is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "wandb",     # For Weights and Biases
        run_name="llava15_7B_mmsd2_01",
        max_length = None,
        # # You MUST put the below items for vision finetuning:
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        dataset_num_proc = 4,
        
    )
)

#@title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

#@title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

print("""### Saving and pushing to hub""") 
model.save_pretrained("llava15_7B_mmsd2_outputs")
model.save_pretrained_merged("llava15_7B_mmsd2_outputs_merged", tokenizer, save_method = "merged_16bit")
# model.push_to_hub("sinngam-khaidem/llava15_7B_mmsd2", token = hf_token) # Online saving
# tokenizer.push_to_hub("sinngam-khaidem/llava15_7B_mmsd2", token = hf_token) # Online saving
# model.push_to_hub_merged("sinngam-khaidem/llava15_7B_mmsd2_merged", tokenizer, save_method = "merged_16bit",token = hf_token)

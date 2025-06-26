from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

text_ncot_df = pd.read_json("/home/gpuuser3/sinngam_albert/work/llava_experiments/generated_annotations/triplets_text_non_cot.json")
text_cot_df = pd.read_json("/home/gpuuser3/sinngam_albert/work/llava_experiments/generated_annotations/triplets_text_cot.json")
image_ncot_df = pd.read_json("/home/gpuuser3/sinngam_albert/work/llava_experiments/generated_annotations/triplets_image_non_cot.json")
image_cot_df = pd.read_json("/home/gpuuser3/sinngam_albert/work/llava_experiments/generated_annotations/triplets_image_cot.json")
mm_ncot_df = pd.read_json("/home/gpuuser3/sinngam_albert/work/llava_experiments/generated_annotations/triplets_mm_non_cot.json")
mm_cot_df = pd.read_json("/home/gpuuser3/sinngam_albert/work/llava_experiments/generated_annotations/triplets_mm_cot.json")

text_ncot_df = text_ncot_df.replace(-1, 0)
text_cot_df = text_cot_df.replace(-1, 0)
image_ncot_df = image_ncot_df.replace(-1, 0)
image_cot_df = image_cot_df.replace(-1, 0)
mm_ncot_df = mm_ncot_df.replace(-1, 0)
mm_cot_df = mm_cot_df.replace(-1, 0)


def compute_performance(df, target_col, pred_col):
    accuracy = accuracy_score(df[target_col], df[pred_col])
    precision = precision_score(df[target_col], df[pred_col], average='macro')
    recall = recall_score(df[target_col], df[pred_col], average='macro')
    f1 = f1_score(df[target_col], df[pred_col], average='macro')
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")

print("LLAVA 1.6 (13B) EVALUATION RESULTS:")
print("--------------------------------------")
print("TEXT MODALITY:")
print("Non COT:-")
compute_performance(text_ncot_df, 'target-text-label', 'llava-text-label')

print("COT:-")
compute_performance(text_cot_df, 'target-text-label', 'llava-text-label')

print("--------------------------------------")
print("IMAGE MODALITY:")
print("Non COT:-")
compute_performance(image_ncot_df, 'target-image-label', 'llava-image-label')

print("COT:-")
compute_performance(image_cot_df, 'target-image-label', 'llava-image-label')

print("--------------------------------------")
print("MULTIMODAL MODALITY:")
print("Non COT:-")
compute_performance(mm_ncot_df, 'target-mm-label', 'llava-mm-label')

print("COT:-")
compute_performance(mm_cot_df, 'target-mm-label', 'llava-mm-label')

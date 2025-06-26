import pandas as pd
import os
import shutil


df = pd.read_json("data/text_json_final/valid.json")

def create_img_path(img_id):
    img_dir = "/Users/sinngamkhaidem/Developer/datasets/MMSD Dataset/dataset_image"
    img_path = f"{img_dir}/{img_id}.jpg"
    return img_path

print(df.columns)
df['image'] = df['image_id'].apply(lambda x: create_img_path(x))

x = 0
for i,row in df.iterrows():
    src = row["image"]
    dest = f"data/dataset_image_test_val/{row['image_id']}.jpg"
    if os.path.exists(src):
        shutil.copy(src, dest)
        x += 1


print(f"Done copying {x} images")   

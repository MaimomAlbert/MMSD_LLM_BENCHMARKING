import pandas as pd
from sklearn.model_selection import train_test_split

# Sample DataFrame
data = pd.read_json("data/text_json_final/valid.json")

# # First, split into train (80%) and temp (20%)
# train_df, temp_df = train_test_split(data, test_size=0.2, random_state=42)

# # Then, split temp into validation (10%) and test (10%)
# val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# # Display sizes
# print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}, Test size: {len(test_df)}")

# train_df.to_json("data/text_json_final/train.json", orient="records", indent=4) 

# test_df.to_json("data/text_json_final/test.json", orient="records", indent=4) 

# val_df.to_json("data/text_json_final/valid.json", orient="records", indent=4) 
print(len(data))


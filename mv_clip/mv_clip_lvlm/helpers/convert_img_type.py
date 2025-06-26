from PIL import Image
import os

def convert_jpg_to_png(jpg_path, png_path):
    image = Image.open(jpg_path)
    image.save(png_path, "PNG")
    os.remove(jpg_path)

# Example usage
# 

for item in os.listdir("data/dataset_image_test_val"):
    jpg_path = f"data/dataset_image_test_val/{item}"
    image = f"{item.split('.')[0]}.png"
    png_path = f"data/dataset_image_test_val/{image}"
    convert_jpg_to_png(jpg_path, png_path)
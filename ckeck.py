from PIL import Image
import os
from collections import defaultdict

def check_image_sizes(data_directory):
    # Dictionary to store counts of each resolution
    resolution_counts = defaultdict(int)
    total_images = 0
    
    # Iterate over all files and subdirectories within the data_directory
    for root, dirs, files in os.walk(data_directory):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                file_path = os.path.join(root, filename)
                try:
                    with Image.open(file_path) as img:
                        width, height = img.size
                        resolution = f"{width}x{height}"
                        resolution_counts[resolution] += 1
                        total_images += 1
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    
    # Print results
    print(f"Total images: {total_images}")
    for resolution, count in resolution_counts.items():
        print(f"Resolution {resolution}: {count} images")

# Example usage
data_directory = r"C:\Personal_Uses\Study_Materials\MP\Final_Work\CP-VTON\data"
check_image_sizes(data_directory)

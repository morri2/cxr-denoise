from PIL import Image
import numpy as np
import sys
import os

# Check command-line arguments
if len(sys.argv) < 2:
    print("Usage: python png_to_npy.py input_image.png [output_array.npy]")
    sys.exit(1)

# Get input and output paths
png_path = sys.argv[1]
output_path = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(png_path)[0] + ".npy"

# Load the image and convert to 8-bit grayscale
img = Image.open(png_path).convert("L")  # "L" mode = 8-bit grayscale

# Convert image to NumPy array (uint8)
img_array = np.array(img, dtype=np.uint8)

# Save the array to .npy file
np.save(output_path, img_array)

print(f"Saved: {output_path}")
print(f"Shape: {img_array.shape}, dtype: {img_array.dtype}")
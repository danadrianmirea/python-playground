from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load the image
input_path = "obstacle.png"  # Replace with your input path
output_path = "shield_pixel_art_updated.png"

# Open image and convert to RGBA
img = Image.open(input_path).convert("RGBA")
img_array = np.array(img)

# Define the yellow RGB value to match
yellow_rgb = (242, 215, 65)

# Define a metallic blue gradient (RGBA)
shield_colors = [
    (150, 200, 255, 255),
    (100, 150, 220, 255),
    (70, 120, 200, 255),
    (40, 90, 160, 255),
]

# Apply gradient to yellow pixels
height = img_array.shape[0]
for y in range(height):
    color = shield_colors[y * len(shield_colors) // height]
    for x in range(img_array.shape[1]):
        r, g, b, a = img_array[y, x]
        if (r, g, b) == yellow_rgb:
            img_array[y, x] = color

# Convert back to image and save
modified_img = Image.fromarray(img_array)
modified_img.save(output_path)

# Optionally display the image
plt.imshow(modified_img)
plt.title("Updated Shield Pixel Art")
plt.axis("off")
plt.show()

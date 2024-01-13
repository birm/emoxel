from PIL import Image
import os
import random

# Set the dimensions for the final collage
collage_width = 500
collage_height = 500

# Set the size for individual images
image_size = 50

# Create a new blank image for the collage
collage = Image.new('RGB', (collage_width, collage_height), (255, 255, 255))

# Path to the directory containing your images
images_directory = './emoji_png'

# Get a list of image files in the directory
image_files = [f for f in os.listdir(images_directory) if f.endswith('.png') or f.endswith('.jpg')]

# Loop through each image and paste it onto the collage at a random position
for image_file in image_files:
    # Open the image
    image_path = os.path.join(images_directory, image_file)
    img = Image.open(image_path)

    # Check if the image has an alpha channel (transparency)
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        # Create a new image with a white background
        img_with_white_bg = Image.new('RGB', img.size, (255, 255, 255))
        img_with_white_bg.paste(img, mask=img.split()[3])  # Use alpha channel as mask
        img = img_with_white_bg

    # Resize the image to a larger size (adjust as needed)
    img = img.resize((image_size, image_size))

    # Generate random x and y positions within the collage
    x_position = random.randint(0, collage_width - image_size)
    y_position = random.randint(0, collage_height - image_size)

    # Paste the resized image onto the collage at the random position
    collage.paste(img, (x_position, y_position))

# Save the final collage
collage.save('figures/chaos_emoji_collage_2.png')

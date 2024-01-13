from PIL import Image
import os

# Set the dimensions for the final collage
collage_width = 1500
collage_height = 1500

# Set the size for individual images
image_size = 50

# Create a new blank image for the collage
collage = Image.new('RGB', (collage_width, collage_height), (255, 255, 255))

# Path to the directory containing your images
images_directory = './emoji_png/'

# Get a list of image files in the directory
image_files = [f for f in os.listdir(images_directory) if f.endswith('.png') or f.endswith('.jpg')]

# Loop through each image and paste it onto the collage
for i, image_file in enumerate(image_files):
    # Open the image
    image_path = os.path.join(images_directory, image_file)
    img = Image.open(image_path)

    # Resize the image to a larger size (adjust as needed)
    img = img.resize((image_size, image_size))

    # Calculate the position to paste the image on the collage
    x_position = (i % (collage_width // image_size)) * image_size
    y_position = (i // (collage_width // image_size)) * image_size

    # Paste the resized image onto the collage
    collage.paste(img, (x_position, y_position))

# Resize the final collage to the desired dimensions
collage = collage.resize((500, 500))

# Save the final collage
collage.save('figures/emojo_collage.png')

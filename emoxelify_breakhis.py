import os
import subprocess
import shutil

def run_emojxel_convert(image_path):
    """Run EmojxelConvert.py for the given image path."""
    emojxel_command = ["python", "EmojxelConvert.py", image_path]
    subprocess.run(emojxel_command)

def run_html_to_png(html_file_path):
    """Run htmlToPng.js for the given HTML file."""
    html_to_png_command = ["node", "htmlToPng.js", html_file_path]
    subprocess.run(html_to_png_command)

def main():
    # Specify the path to your original images
    #already did "./breakhis/benign-adenosis/", 
    original_image_paths = ['./breakhis/malignant-ductal_carcinoma', './breakhis/benign-fibroadenoma', './breakhis/malignant-lobular_carcinoma']

    for original_image_path in original_image_paths:

        # Specify the path where the HTML files will be saved
        html_output_path = original_image_path
        
        # Create output directories if they don't exist
        os.makedirs(html_output_path, exist_ok=True)

        # Process each original image
        for image_filename in os.listdir(original_image_path):
            if image_filename.endswith(".jpg") or image_filename.endswith(".png"):
                image_path = os.path.join(original_image_path, image_filename)

                # Run EmojxelConvert.py to generate HTML file
                html_output_file = os.path.join(html_output_path, f"{image_filename}.html")
                run_emojxel_convert(image_path)

                # Run htmlToPng.js to convert HTML to PNG
                run_html_to_png(html_output_file)

if __name__ == "__main__":
    main()

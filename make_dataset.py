import os
import subprocess

def run_emojxel_convert(image_path):
    """Run EmojxelConvert.py for the given image path."""
    emojxel_command = ["python", "EmoxelConvert.py", image_path]
    subprocess.run(emojxel_command)

def run_html_to_png(html_file_path):
    """Run htmlToPng.js for the given HTML file."""
    html_to_png_command = ["node", "htmlToPng.js", html_file_path]
    subprocess.run(html_to_png_command)

def main():
    # Specify the path to your original images
    original_image_path = "/path/to/original/images"

    # Specify the path where the HTML files will be saved
    html_output_path = "./html_output"

    # Specify the path where the PNG files will be saved
    png_output_path = "./screenshots"

    # Create output directories if they don't exist
    os.makedirs(html_output_path, exist_ok=True)
    os.makedirs(png_output_path, exist_ok=True)

    # Process each original image
    for image_filename in os.listdir(original_image_path):
        if image_filename.endswith(".jpg") or image_filename.endswith(".png"):
            image_path = os.path.join(original_image_path, image_filename)

            # Run EmoxelConvert.py to generate HTML file
            html_output_file = os.path.join(html_output_path, f"{image_filename}.html")
            run_emojxel_convert(image_path)

            # Run htmlToPng.js to convert HTML to PNG
            run_html_to_png(html_output_file)

            # Move the resulting PNG file to the screenshots directory
            png_output_file = os.path.join(png_output_path, f"{image_filename}.png")
            shutil.move(f"{image_filename}.png", png_output_file)

if __name__ == "__main__":
    main()

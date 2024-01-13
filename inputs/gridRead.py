import os

def create_html_grid(input_directory):
    # Get a list of text files in the input directory
    text_files = [file for file in os.listdir(input_directory) if file.endswith(".txt")]
    for text_file in text_files:
        # Open the output HTML file for writing
        with open(text_file.split('.')[0]+".html", "w", encoding="utf-8") as html_file:
            # Write the HTML header
            html_file.write("<!DOCTYPE html>\n<html>\n<head>\n<meta charset='utf-8'>\n</head>\n<body>\n")
            html_file.write("<div style='font-size: 24px;'>\n")

            # Read lines from the text file and write them to the HTML file
            with open(os.path.join(input_directory, text_file), "r", encoding="utf-8") as file:
                for line in file:
                    # Add padding to ensure each displayed emoji/character has 32px width
                    padded_line = line.rstrip().ljust(32, ' ')  # Use a regular space for padding
                    html_file.write(padded_line + "<br>")

            html_file.write("</div>\n")

            # Write the HTML footer
            html_file.write("</body>\n</html>")

if __name__ == "__main__":
    input_directory = "./"
    create_html_grid(input_directory)

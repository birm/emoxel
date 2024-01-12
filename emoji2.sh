#!/bin/bash

input_directory="emoji_html"
output_directory="emoji_png"
font_path="/path/to/your/font.ttf"  # Replace with the actual path to your font file

# Create the output directory if it doesn't exist
mkdir -p "$output_directory"

# Iterate over each HTML file in the input directory
for html_file in "$input_directory"/*.html; do
    # Extract the file name without extension
    file_name=$(basename -- "$html_file")
    file_name_no_extension="${file_name%.*}"

    # Define the output file path
    output_file="$output_directory/$file_name_no_extension.png"

    # Use convert from ImageMagick to convert HTML to PNG with font
    convert "$html_file" \
        -background white \
        -flatten \
        -font "$font_path" \
        -pointsize 32 \
        label:@- "$output_file"

    echo "Converted $html_file to $output_file"
done

#!/bin/bash

# Create a directory to store the images and CSV file
output_directory="emoji"
mkdir -p "$output_directory"

# Get the list of commonly used emojis
common_emojis=("😀" "😂" "😍" "🔥")  # Add more as needed

# Specify the NotoColorEmoji font path
noto_font_path='./NotoColorEmoji.ttf'

# Create and save PNG images along with CSV data
csv_file_path="emoji_data.csv"
openssl base64 -A -in "$noto_font_path" >> "$csv_file_path"
echo ",$output_path,$emoji_unicode," >> "$csv_file_path"


for ((i=0; i<${#common_emojis[@]}; i++)); do
    emoji_unicode="${common_emojis[i]}"
    output_path="${output_directory}/$((i+1))_${emoji_unicode}.png"

    # Create HTML content with emoji and specify the font
    html_content="<div style=\"font-size:32px; font-family: NotoColorEmoji;\">${emoji_unicode}</div>"

    # Generate PNG image using wkhtmltoimage
    wkhtmltoimage \
        --format png \
        --quiet \
        --no-images \
        --custom-header "User-Agent" "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36" \
        --user-style-sheet <(echo '@font-face { font-family: "NotoColorEmoji"; src: url(data:font/truetype;base64,'$(base64 -w 0 "$noto_font_path")'); }') \
        <(echo "$html_content") "$output_path"

    # Append to CSV data
    echo "$output_path,$emoji_unicode," >> "$csv_file_path"
done

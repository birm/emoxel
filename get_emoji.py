import os
import csv
import emoji

# Create a directory to store HTML files
output_directory = "emoji_html"
os.makedirs(output_directory, exist_ok=True)

# Get the list of commonly used emojis
common_emojis = emoji.EMOJI_DATA.keys()

# Create HTML files
html_data = []

for i, unicode in enumerate(common_emojis):
    # Create HTML content
    html_content = f'<meta charset="utf-8"><div style="font-size: 32px;position: absolute;left:-2;top:-2;">{unicode}</div>'

    # Save HTML content to a file
    html_filename = f"{i + 1}_{unicode}.html"
    html_path = os.path.join(output_directory, html_filename)
    with open(html_path, 'w') as html_file:
        html_file.write(html_content)

    # Append to HTML data
    html_data.append((html_filename, unicode))

# Save CSV file
csv_file_path = "emoji_html/emoji_html_data.csv"
with open(csv_file_path, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['filename', 'unicode'])
    csv_writer.writerows(html_data)

print(f"HTML files and CSV file saved in '{output_directory}' and '{csv_file_path}' respectively.")

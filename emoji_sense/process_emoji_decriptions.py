import json
# import unicodedata
from transformers import pipeline 

def concatenate_senses_text(senses):
    text = ""
    for key, value in senses.items():
        if isinstance(value, list):
            for entry in value:
                if isinstance(entry, dict):
                    text += concatenate_senses_text(entry)
                elif isinstance(entry, str):
                    text += entry + ' '
        elif isinstance(value, dict):
            text += " " + concatenate_senses_text(value)
    return text.strip()

def process_data(json_data):
    concatenated_data = {}
    for emoji_record in json_data:
        unicode_key = emoji_record.get('unicode', '')
        emoji_character = chr(int(unicode_key.split()[0][2:], 16))  # Convert Unicode to emoji character
        # print(emoji_character)
        # normalized_emoji_character = unicodedata.normalize('NFC', emoji_character)  # Normalize Unicode character
        senses = emoji_record.get('senses', {})
        concatenated_text = concatenate_senses_text(senses)
        concatenated_data[emoji_character] = concatenated_text
    return concatenated_data

# Read data from data.json
with open('emojis.json', 'r') as json_file:
    data = json.load(json_file)

# Process the data
result = process_data(data)
print(result.keys())



# testing sentiment
emoji_character = '😴'  # Replace with the specific emoji character you want to access
if emoji_character in result:
    senses_text = result[emoji_character]
    print(f"Senses Text for {emoji_character}: {senses_text}")
    sentiment_analysis_pipeline = pipeline("sentiment-analysis")
    # Example text from the "senses" field
    # Analyze sentiment
    sentiment_result = sentiment_analysis_pipeline(senses_text)

    # Print sentiment result
    print(sentiment_result)
else:
    print(f"The key {emoji_character} is not present in the dictionary.")

with open("emojiSenseMap.json", 'w') as f:
    json.dump(result, f)


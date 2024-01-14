import json
from transformers import pipeline 

# load the emojiSenseMap

def emojiSense(map, emoji):
    emojiSenseMap = json.load(map) # emojiSenseMap.json file pointer
    senses_text = emojiSenseMap.get(emoji, False)
    if not senses_text:
        return 0.0
    sentiment_analysis_pipeline = pipeline("sentiment-analysis")
    sentiment_result = sentiment_analysis_pipeline(senses_text)[0]
    if sentiment_result['label'] == "NEGATIVE":
        return -1 * sentiment_result['score']
    else:
        return sentiment_result['score']

if __name__ == "__main__":
    emoji = '🍕'
    with open('./emojiSenseMap.json', 'r') as f:
        print(emojiSense(f, emoji))
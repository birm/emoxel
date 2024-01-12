import cv2
import numpy as np
import os
import sys

def extract_features(image):
    if image is None:
        raise ValueError("Error loading image")

    if len(image.shape) == 3:  # Convert to grayscale if the image has 3 channels
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Average Intensity
    avg_intensity = np.mean(gray_image)

    # Edge Density (Canny Edge Detector)
    edges = cv2.Canny(gray_image, 50, 150)
    edge_density = np.sum(edges) / (gray_image.shape[0] * gray_image.shape[1])

    # Entropy
    _, gray_image_binary = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
    pixel_counts = cv2.countNonZero(gray_image_binary)
    epsilon = 1e-10
    image_entropy = -np.sum((pixel_counts + epsilon) / (gray_image.shape[0] * gray_image.shape[1]) * np.log2((pixel_counts + epsilon) / (gray_image.shape[0] * gray_image.shape[1])))
    return np.concatenate([[avg_intensity], [edge_density, image_entropy]])

def calculate_feature_similarity(query_image, candidate_images):
    query_features = extract_features(query_image)

    similarities = []
    for candidate_image in candidate_images:
        image_data = candidate_image['data']
        candidate_features = extract_features(image_data)
        # Calculate Euclidean distance as a similarity measure
        similarity_score = np.linalg.norm(query_features - candidate_features)

        similarities.append({'emoji': candidate_image['emoji'], 'score': similarity_score})

    return similarities


if __name__ == "__main__":
    candidates_directory = "./emoji_png/"
    matching_filename = sys.argv[1]
    
    # Example usage:
    query_image = cv2.imread(matching_filename, cv2.IMREAD_GRAYSCALE)

    candidate_images = [{'emoji': file, 'data': cv2.imread(os.path.join(candidates_directory, file), cv2.IMREAD_GRAYSCALE)} for file in os.listdir(candidates_directory) if file.endswith(".png") and not file == matching_filename]

    chunk_size = 32
    image_height, image_width = query_image.shape
    chunks = []

    text_out = ""

    for y in range(0, image_height, chunk_size):
        for x in range(0, image_width, chunk_size):
            chunk = query_image[y:y+chunk_size, x:x+chunk_size]
            chunks.append((y, y+chunk_size, x, x+chunk_size))

            # Extract features and find the closest match for each chunk
            chunk_similarity = calculate_feature_similarity(chunk, candidate_images)
            closest_match = min(chunk_similarity, key=lambda x: x['score'])
            if np.isnan(closest_match['score']):
                exit(-1)
            print(x, y, closest_match)
            text_out += closest_match['emoji'].split("_")[1].split(".")[0]
        text_out += "\n"

    with open(matching_filename + ".txt", 'w') as f:
        f.write(text_out)
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the TFRecord dataset
tfrecord_path = "./tf_dataset.tfrecord"

def _parse_function(proto):
    keys_to_features = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    parsed_features = tf.io.parse_single_example(proto, keys_to_features)
    parsed_features['image'] = tf.io.decode_jpeg(parsed_features['image'], channels=3)
    return parsed_features['image'], parsed_features['label']

tf_dataset = tf.data.TFRecordDataset(tfrecord_path)
tf_dataset = tf_dataset.map(_parse_function)

# Visualize some images with their actual labels
num_images_to_visualize = 10

label_mapping = {0:'benign-adenosis', 1: 'benign-fibroadenoma', 2: 'malignant-ductal_carcinoma', 3: 'malignant-lobular_carcinoma'}

for image, label in tf_dataset.take(num_images_to_visualize):
    # Convert to NumPy arrays
    image = image.numpy()
    label = int(label)

    # Display the image with the actual label
    plt.imshow(image)
    plt.title(f"Label: {label_mapping[label]}")
    plt.show()

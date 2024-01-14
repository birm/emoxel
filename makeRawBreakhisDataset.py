import os
import tensorflow as tf
from PIL import Image
import numpy as np

def preprocess_image(image_path):
    """Preprocess image by grabbing the top-left 256x256 image with a 10x10 offset."""
    image = Image.open(image_path)
    cropped_image = image.crop((10, 10, 266, 266))  # Crop to get 256x256 image with 10x10 offset
    return np.array(cropped_image)

def create_dataset(dataset_path):
    """Create a TensorFlow dataset."""
    image_labels = os.listdir(dataset_path)
    images = []
    labels = []

    for label in image_labels:
        label_path = os.path.join(dataset_path, label)

        if os.path.isdir(label_path):
            for filename in os.listdir(label_path):
                if filename.endswith(".png"):
                    if not filename.endswith("converted.png"):
                        image_path = os.path.join(label_path, filename)

                        # Preprocess image
                        preprocessed_image = preprocess_image(image_path)

                        # Add image and label to the dataset
                        images.append(preprocessed_image)
                        labels.append(label)

    return np.array(images), np.array(labels)

def main():
    # Specify the path to the preprocessed images
    dataset_path = "./breakhis"

    # Create the TensorFlow dataset
    images, labels = create_dataset(dataset_path)
    print(labels)

    # Convert labels to one-hot encoding
    label_mapping = {label: index for index, label in enumerate(np.unique(labels))}
    print(label_mapping)
    labels_one_hot = np.array([tf.one_hot(label_mapping[label], len(label_mapping)) for label in labels])
    # Create TensorFlow dataset
    tf_dataset = tf.data.Dataset.from_tensor_slices((images, labels_one_hot))

    # Shuffle and batch the dataset
    tf_dataset = tf_dataset.shuffle(buffer_size=len(images)).batch(batch_size=1)

    # Save the dataset as TFRecord (optional)
    tf_record_path = "./breakhisRaw.tfrecord"

    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        value = Image.fromarray(value).convert("RGB")
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(np.array(value)).numpy()]))

    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    with tf.io.TFRecordWriter(tf_record_path) as writer:
        i = 0
        for image, label in tf_dataset:
            i+=1
            feature = {
                'image': _bytes_feature(image[0].numpy()),
                'label': _int64_feature(np.argmax(label[0].numpy()))
            }
            tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(tf_example.SerializeToString())
        print("wrote", i, "records")
if __name__ == "__main__":
    main()

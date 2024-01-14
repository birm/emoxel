import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import struct
from tensorflow.keras import layers
import EmoxelConvert

class EmojiConvertLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(EmojiConvertLayer, self).__init__()

    def build(self, input_shape):
        pass

    def call(self, inputs):
        print("Input shape to EmojiConvertLayer:", inputs.shape)  # Print input shape for confirmation

        # Resize input images to 32x32
        resized_regions = tf.image.resize(inputs, [32, 32])

        # Convert to a natural number using the provided function
        natural_number = tf.keras.layers.Lambda(self.convert_to_number, input_shape=(32, 32, 1))(resized_regions)

        # Add an extra dimension to the output
        return tf.expand_dims(natural_number, axis=-1)

    def compute_output_shape(self, input_shape):
        # Assuming batch size is the first dimension
        return tf.TensorShape((input_shape[0],))

    def convert_to_number(self, resized_regions):
        # Apply your EmoxelConvert function to get the Unicode of the closest emoji
        # Note: Ensure that EmoxelConvert.toUnicode can handle batch processing
        emoji_unicode = tf.py_function(lambda x: unicode_to_float(EmoxelConvert.toUnicode(tensors_to_image(x))), [resized_regions], tf.float32)

        # Set a static shape for the tensor
        emoji_unicode.set_shape([])

        # Explicitly cast to float32
        emoji_float = tf.cast(emoji_unicode, tf.float32)

        return emoji_float



def unicode_to_float(unicode_string):
    # Convert Unicode string to bytes
    utf8_bytes = unicode_string.encode('utf-8')

    # Pad the byte string with zeros if its length is less than 4
    utf8_bytes_padded = utf8_bytes.ljust(4, b'\x00')

    # Take the first four bytes and unpack them as a float
    float_value = struct.unpack('f', utf8_bytes_padded[:4])[0]

    return float(float_value)  # Explicitly cast to float

def tensors_to_image(tensor):
    # Assuming tensor is a tf.Tensor with shape (32, 32, 1)

    # Ensure the tensor has the correct shape
    tensor = tf.reshape(tensor, [-1, 32, 32, 1])

    # Normalize pixel values to [0, 255]
    normalized_tensor = (tensor - tf.reduce_min(tensor)) / (tf.reduce_max(tensor) - tf.reduce_min(tensor)) * 255

    # Convert to NumPy array
    image_array = tf.keras.backend.eval(normalized_tensor)

    return image_array[0, :, :, 0].astype(np.uint8)

# Load the TFRecord dataset
tfrecord_path = "./breakhisRaw.tfrecord"

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

# Split the dataset into training and validation sets
images, labels = zip(*tf_dataset)
images = np.array(images)
labels = np.array(labels)

X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42069)

model = tf.keras.Sequential([
    # Convolutional layers
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),  # Output shape: (254, 254, 32)
    layers.MaxPooling2D((2, 2)),  # Output shape: (127, 127, 32)
    layers.Conv2D(64, (3, 3), activation='relu'),  # Output shape: (125, 125, 64)
    layers.MaxPooling2D((2, 2)),  # Output shape: (62, 62, 64)
    layers.Conv2D(128, (27, 27), activation='relu'),  # Output shape: (36, 36, 128)
    # Sum along the third axis to get (36, 36, 1)
    layers.Lambda(lambda x: tf.reduce_sum(x, axis=-1, keepdims=True)),  # Output shape: (36, 36, 1)
    EmojiConvertLayer(),
    layers.Flatten(),
    layers.Dense(4, activation='softmax'),
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

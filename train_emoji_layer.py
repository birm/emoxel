import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import EmoxelConvert
import struct
import matplotlib.pyplot as plt
from torchinfo import summary as torchinfo_summary

# Assuming you have your data loaded in PyTorch tensors X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor
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
labels = tf.keras.utils.to_categorical(labels, num_classes=4)  # Convert labels to one-hot encoding
labels = np.array(labels)

X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42069)

def showMe(image, title="Image"):
    """
    Display the image using Matplotlib.

    Parameters:
    - image: NumPy array representing the image.
    - title: Title of the displayed image (default is "Image").
    """
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()

class EmoxelLayer(nn.Module):
    def forward(self, x):
        return emoxelify(x)


def unicode_to_float(unicode_string):
    # Convert Unicode string to bytes
    utf8_bytes = unicode_string.encode('utf-8')

    # Pad the byte string with zeros if its length is less than 4
    utf8_bytes_padded = utf8_bytes.ljust(4, b'\x00')

    # Take the first four bytes and unpack them as a float
    float_value = struct.unpack('f', utf8_bytes_padded[:4])[0]

    return float(float_value)  # Explicitly cast to float

def emoxelify(x):
    # Assuming x has shape (batch_size, channels, height, width)
    normalized_tensor = torch.nn.functional.normalize(x, p=2, dim=(2, 3))
    
    # Reshape to (batch_size, height, width, channels) for compatibility with EmoxelConvert
    numpy_array = normalized_tensor.permute(0, 2, 3, 1).detach().cpu().numpy()
    
    # Ensure the last dimension is 1
    numpy_array = numpy_array[0,:,:,0:3]
    # Adjust contrast of the image with center at 128
 
    # Calculate the first and fourth quartiles
    first_quartile = np.percentile(numpy_array, 45)
    fourth_quartile = np.percentile(numpy_array, 55)

    # Adjust contrast of the image
    contrast_factor = 255 / (fourth_quartile - first_quartile)

    numpy_array = (numpy_array - first_quartile) * contrast_factor

    # Apply thresholding to enhance edges/features for each pixel
    for i in range(3):  # Loop over RGB channels
        channel_values = numpy_array[:, :, i]
        numpy_array[ (channel_values > 200)] = [255, 255, 255]  # Set entire pixel to black or white

    # Ensure values are in [0, 255]
    numpy_array = np.clip(numpy_array, 0, 255).astype(np.uint8)



    numpy_array = numpy_array.astype(np.uint8)
    # need something to make numpy_array like an image for the cv functions within toUnicode
    
    #showMe(numpy_array)
    unicode_strings = EmoxelConvert.toUnicode(numpy_array)
    float_value = unicode_to_float(unicode_strings)
    return float_value


# Convert NumPy arrays to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train).permute(0, 3, 1, 2).float()
y_train_tensor = torch.from_numpy(y_train)
X_val_tensor = torch.from_numpy(X_val).permute(0, 3, 1, 2).float()
y_val_tensor = torch.from_numpy(y_val)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

class CustomModel(nn.Module):
    def __init__(self, input_shape=(3, 256, 256), num_classes=4):
        super(CustomModel, self).__init__()

        # Additional layers for feature extraction
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        # EmoxelLayer
        self.emoxel = EmoxelLayer()

        # Layers for handling scalar input
        self.scalar_fc = nn.Linear(1, 256)  # Assuming input is a scalar tensor
        self.scalar_dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = self.pool(x)

        # EmoxelLayer
        x = torch.as_tensor(self.emoxel(x))

        # Handling scalar input
        x = torch.relu(self.scalar_fc(x.view(-1, 1)))  # Reshape to handle scalar input
        x = self.scalar_dropout(x)
        x = self.fc2(x)

        return x



# Instantiate the model
custom_model = CustomModel()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(custom_model.parameters(), lr=0.001)

# tl
# Training loop
num_epochs = 20

# Initialize accuracy lists
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    custom_model.train()
    correct_train = 0
    total_train = 0
    train_loss = 0.0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = custom_model(inputs)

        # Check if outputs is None
        if outputs is None:
            print("Forward pass returned None. Check the model's forward method.")
            continue

        # Ensure labels have the correct shape
        if labels.dim() == 1:
            labels = labels.unsqueeze(1)

        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels.argmax(dim=1)).sum().item()
        print(predicted, labels, labels.argmax(dim=1))

        loss = criterion(outputs, labels.argmax(dim=1))  # Use argmax to get class indices
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Compute training accuracy and loss
    train_accuracy = correct_train / total_train
    avg_train_loss = train_loss / len(train_loader)

    # Log training accuracy and loss
    train_accuracies.append(train_accuracy)

    # Evaluation
    custom_model.eval()
    correct_val = 0
    total_val = 0
    val_loss = 0.0

    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = custom_model(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels.argmax(dim=1)).sum().item()

            val_loss += criterion(outputs, labels.argmax(dim=1))

    # Compute validation accuracy and loss
    val_accuracy = correct_val / total_val
    avg_val_loss = val_loss / len(val_loader)

    # Log validation accuracy and loss
    val_accuracies.append(val_accuracy)

    # Print training and validation metrics
    print(f'Epoch {epoch + 1}/{num_epochs}: '
          f'Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.4f} | '
          f'Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

# Plot the training and validation accuracies
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Training Accuracy')
plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(0, 1)  # Ensure y-axis is within [0, 1]
plt.legend()
plt.show()

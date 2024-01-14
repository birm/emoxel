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


class SpecialMeowLayer(nn.Module):
    def forward(self, x):
        return specialMeow(x)


def unicode_to_float(unicode_string):
    # Convert Unicode string to bytes
    utf8_bytes = unicode_string.encode('utf-8')

    # Pad the byte string with zeros if its length is less than 4
    utf8_bytes_padded = utf8_bytes.ljust(4, b'\x00')

    # Take the first four bytes and unpack them as a float
    float_value = struct.unpack('f', utf8_bytes_padded[:4])[0]

    return float(float_value)  # Explicitly cast to float

def specialMeow(x):
    # Assuming x has shape (batch_size, channels, height, width)
    normalized_tensor = torch.nn.functional.normalize(x, p=2, dim=(2, 3))
    
    # Reshape to (batch_size, height, width, channels) for compatibility with EmoxelConvert
    numpy_array = normalized_tensor.permute(0, 2, 3, 1).detach().cpu().numpy()
    
    # Ensure the last dimension is 1
    numpy_array = numpy_array[0,:,:,0:3]
    numpy_array = ((numpy_array - np.min(numpy_array)) / (np.max(numpy_array) - np.min(numpy_array)) * 128 )+ 128
    numpy_array = numpy_array.astype(np.uint8)
    # need something to make numpy_array like an image for the cv functions within toUnicode
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

        # SpecialMeowLayer
        self.special_meow = SpecialMeowLayer()

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

        # SpecialMeowLayer
        x = torch.as_tensor(self.special_meow(x))

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
train_losses = []
val_losses = []

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    custom_model.train()
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

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Log training loss
    train_losses.append(loss.item())

    # Evaluation
    custom_model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for inputs, labels in val_loader:
            outputs = custom_model(inputs)
            val_loss += criterion(outputs, labels).item()

        # Log validation loss
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

# Plot the training and validation losses
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

torchinfo_summary(custom_model, input_size=(1, 3, 256, 256))  # Adjust input_size as needed
print("done")


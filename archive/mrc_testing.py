import mrcfile
import numpy as np
import os
import pathlib
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense

X_train = np.random.rand(100, 10)  # 100 samples with 10 features each
y_train = np.random.randint(0, 2, size=(100,))  # Binary labels

# print(X_train)
# print(y_train)

# print(os.getcwd())
# with mrcfile.open('mrc/atlas-mrc/bltp2/atlas_BLTP2_0.mrc') as mrc:
#     arr = mrc.data
#     print(f"Dimensions: {arr.shape}")
#     print(arr[0][0])
    
def load_mrc_files_from_directory(directory):
    mrc_files = []
    labels = []

    # Iterate over subdirectories (classes)
    class_names = sorted(os.listdir(directory))
    for class_index, class_name in enumerate(class_names):
        class_dir = os.path.join(directory, class_name)
        if os.path.isdir(class_dir):
            # Iterate over .mrc files in the current class directory
            for filename in sorted(os.listdir(class_dir)):
                if filename.endswith('.mrc'):
                    file_path = os.path.join(class_dir, filename)
                    # Load .mrc file
                    with mrcfile.open(file_path) as mrc:
                        mrc_data = mrc.data     # alternatively: .astype(np.float32)
                        mrc_files.append(mrc_data)
                        labels.append(class_index)

    return mrc_files, labels, class_names

def mrc_dataset_from_directory(directory, batch_size=32, validation_split=None, seed=123):
    # Load .mrc files from directory
    mrc_files, labels, class_names = load_mrc_files_from_directory(directory)

    # Convert lists to TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((mrc_files, labels))

    if validation_split:
        # Determine sizes of training and validation sets
        num_samples = len(mrc_files)
        num_validation_samples = int(validation_split * num_samples)
        num_training_samples = num_samples - num_validation_samples

        # Shuffle and split dataset into training and validation sets
        dataset = dataset.shuffle(num_samples, seed=seed)
        train_dataset = dataset.take(num_training_samples)
        validation_dataset = dataset.skip(num_training_samples)
        validation_dataset = validation_dataset.take(num_validation_samples)

        return train_dataset, validation_dataset, class_names
    else:
        # Shuffle and batch the dataset
        dataset = dataset.shuffle(len(mrc_files), seed=seed).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        return dataset, class_names





data_dir = "mrc/atlas-mrc"
data_dir = pathlib.Path(data_dir)

batch_size = 32
img_height = img_width = img_depth = 100
    
# train_ds, val_ds, class_names = mrc_dataset_from_directory(
#     data_dir,
#     batch_size=batch_size,
#     validation_split=0.2,
#     seed=123
# )

# class_names = sorted(os.listdir(data_dir))
# print(class_names)
# for class_index, class_name in enumerate(class_names):
#     class_dir = os.path.join(data_dir, class_name)
#     print(class_dir)

mrc_files, labels, class_names = load_mrc_files_from_directory(data_dir)
# print(mrc_files[0].shape)
# print(len(labels))
# print(class_names)

# print(train_ds.shape)
# print(val_ds.shape)
# print(class_names)

# Convert list of 3D numpy arrays to a single 4D numpy array
# Then, convert list of labels to a numpy array
x = np.array(mrc_files)
y = np.array(labels)

# Preprocessing (e.g., normalization)
x_normalized = x / 255.0

# Shuffle data
indices = np.arange(x.shape[0])
x_shuffled = x_normalized[indices]
y_shuffled = y[indices]

# Split data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_shuffled, y_shuffled, test_size=0.2, random_state=42)

print(f"Training Dataset: {len(x_train)} tomograms and {len(y_train)} labels")
print(f"Validation Dataset: {len(x_val)} tomograms and {len(y_val)} labels")

# print(f"Training Peek:\n{x_train[0]}\n{y_train[0]}")
# print(f"Validation Peek:\n{x_val[0]}\n{y_val[0]}")

# print(y_train)
# print(y_val)

num_classes = 3

model = Sequential([
    Conv3D(32, kernel_size=(3, 3, 3), activation='relu', input_shape=(100, 100, 100, 1)),
    MaxPooling3D(pool_size=(2, 2, 2)),
    Conv3D(64, kernel_size=(3, 3, 3), activation='relu'),
    MaxPooling3D(pool_size=(2, 2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

epochs = 4
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val))

x_test = x_val[0]
y_test = y_val[0]

test_loss, test_accuracy = model.evaluate(x_test, y_test)
print('Test loss:', test_loss)
print('Test accuracy:', test_accuracy)

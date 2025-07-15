import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from keras.datasets import cifar10

# Load and preprocess the dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize pixel values
x_train, x_test = x_train.reshape(-1, 32 * 32 * 3), x_test.reshape(-1, 32 * 32 * 3)  # Flatten images
y_train, y_test = y_train.flatten(), y_test.flatten()

# Filter CIFAR-10 trucks (class 9)
truck_class = 0
train_indices_cifar = (y_train == 9)
test_indices_cifar = (y_test == 9)

# Load military truck images
loaded_images = np.load("cifar_like_truck_images_1860.npy")
print("Loaded military truck images shape:", loaded_images.shape)

# Assign class labels
x_train_trucks = x_train[train_indices_cifar][:1860]
y_train_trucks = np.full(x_train_trucks.shape[0], truck_class)  # Label as truck class
x_test_trucks = x_test[test_indices_cifar][:1860]
y_test_trucks = np.full(x_test_trucks.shape[0], truck_class)  # Label as truck class

x_train_military = loaded_images
y_train_military = np.full(x_train_military.shape[0], 1)  # Label as military truck class
x_test_military = loaded_images
y_test_military = np.full(x_test_military.shape[0], 1)  # Label as military truck class

# Combine data
x_train_combined = np.vstack((x_train_trucks, x_train_military))
y_train_combined = np.hstack((y_train_trucks, y_train_military))
x_test_combined = np.vstack((x_test_trucks, x_test_military))
y_test_combined = np.hstack((y_test_trucks, y_test_military))

# Shuffle data
shuffled_indices = np.random.permutation(x_train_combined.shape[0])
x_train_combined = x_train_combined[shuffled_indices]
y_train_combined = y_train_combined[shuffled_indices]

# Visualize all military trucks in a grid
def plot_military_trucks(images, num_images_per_row=4):
    num_images = 16#images.shape[0]
    num_rows = (num_images + num_images_per_row - 1) // num_images_per_row  # Calculate number of rows
    
    plt.figure(figsize=(num_images_per_row * 1.5, num_rows * 1.5))
    for i in range(num_images):
        plt.subplot(num_rows, num_images_per_row, i + 1)
        plt.imshow(images[i])
        plt.axis('off')  # Turn off axis
    plt.tight_layout()
    plt.show()
    
# The loaded_images should be an array of shape (num_images, 32, 32, 3)
#plot_military_trucks(loaded_images.reshape(-1, 32, 32, 3))
plot_military_trucks(x_train_combined.reshape(-1, 32, 32, 3))
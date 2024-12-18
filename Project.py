import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import shuffle

# Load dataset
data_set = pd.read_csv("A_Z Handwritten Data.csv", header=None)  # header=None as the data has no header
print(data_set.describe())
data_shuffled = shuffle(data_set)


def display_image_grid(flattened_vectors, image_size=(28, 28), grid_size=(3, 3)):    
    # Create a figure with a grid of subplots
    plt.subplots(grid_size[0], grid_size[1], figsize=(8, 8))
    
    # Plot images
    for i in range(grid_size[0] * grid_size[1]):
        if i < len(flattened_vectors):
            plt.subplot(grid_size[0], grid_size[1], i + 1)
            plt.imshow(flattened_vectors[i].reshape(image_size), cmap="Greys")
            plt.axis("off")
        else:
            plt.axis("off")
    
    # Adjust layout
    plt.tight_layout()
    plt.show()


def data_exploration():
    # Check missing values
    print(f"Missing values: {data_shuffled.isna().sum().sum()}")
    
    # Separate features (pixel values) and labels (alphabets)
    y_data = data_shuffled.iloc[:, 0]   # All rows, only the first column (labels)
    x_data = data_shuffled.iloc[:, 1:]  # All rows, all columns except the first (784 pixels in row)

    # Identify the number of unique classes
    unique_classes = np.unique(y_data)

    # Number of unique classes
    print(f"Number of unique classes: {len(unique_classes)}")

    # Show the distribution of the labels
    sns.countplot(x=y_data)  # x-axis will be y (the labels)
    plt.title("Distribution of The Labels")
    plt.xlabel("Alphabet")
    plt.ylabel("Frequency")
    plt.show()

    # Normalize the data
    x_data = x_data / 255.0 

    # Display the first 6x6 images
    display_image_grid(x_data.values, grid_size=(6, 6))

    return x_data, y_data
    

if __name__ == "__main__":
    x_data, y_data = data_exploration()
    # train test split
    # SVM
    # Logistic Regression
    # NN
    
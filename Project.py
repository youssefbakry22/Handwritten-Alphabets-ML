import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.utils import shuffle
import tensorflow as tf


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
    print("\n\n-----------------Data Exploration-----------------")
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


def SVM_Linear(x_train, x_test, y_train, y_test):
    print("-----------------SVM Linear-----------------")
    # Create the model
    model = SVC(kernel="linear")
    
    # Train the model
    model.fit(x_train, y_train)
    
    # Predict
    y_pred = model.predict(x_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(f1_score(y_test, y_pred, average="weighted"))


def SVM_NonLinear(x_train, x_test, y_train, y_test):
    print("\n\n-----------------SVM Non-Linear (RBF)-----------------")
    # Create the model
    model = SVC(kernel="rbf")
    
    # Train the model
    model.fit(x_train, y_train)
    
    # Predict
    y_pred = model.predict(x_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(f1_score(y_test, y_pred, average="weighted"))


def NN_1(x_train, x_test, y_train, y_test):
    print("\n\n-----------------Neural Network 1-----------------")
    # Create the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(26, activation="softmax")
    ])

    # Compile the model
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Train the model
    model.fit(x_train, y_train, epochs=10)

    return model


def NN_2(x_train, x_test, y_train, y_test):
    print("\n\n-----------------Neural Network 2-----------------")
    # Create the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', Input=(28, 28, 1)),  # Convolutional layer
        tf.keras.layers.MaxPooling2D((2, 2)),                                           # Down-sampling
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),                                                     # Flatten to connect to Dense layer
        tf.keras.layers.Dense(128, activation='relu'),                                 # Hidden layer
        tf.keras.layers.Dense(26, activation='softmax')                                # Output layer
    ])

    # Compile the model
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Train the model
    model.fit(x_train, y_train, epochs=10)

    return model


if __name__ == "__main__":
    x_data, y_data = data_exploration()
    # train test split
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)
    # SVM
    # SVM_Linear(x_train, x_test, y_train, y_test)
    # SVM_NonLinear(x_train, x_test, y_train, y_test)

    # Logistic Regression
    # NN
    model_1 = NN_1(x_train, x_test, y_train, y_test)
    # model_2 = NN_2(x_train, x_test, y_train, y_test)

    # save the best model
    model_1.save("model_1.keras")
    # model_2.save("model_2.keras")
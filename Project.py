import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.utils import shuffle
import tensorflow as tf

# might need this to show the outputs
alpabet_dict = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "E",
    5: "F",
    6: "G",
    7: "H",
    8: "I",
    9: "J",
    10: "K",
    11: "L",
    12: "M",
    13: "N",
    14: "O",
    15: "P",
    16: "Q",
    17: "R",
    18: "S",
    19: "T",
    20: "U",
    21: "V",
    22: "W",
    23: "X",
    24: "Y",
    25: "Z",
}

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
    y_data = data_shuffled.iloc[:, 0]  # All rows, only the first column (labels)
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


def NN_1(x_train, x_validate, y_train, y_validate):
    print("\n\n-----------------Neural Network 1-----------------")
    # Create the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),      # Flatten the input
        tf.keras.layers.Dense(128, activation="relu"),      # Hidden layer
        tf.keras.layers.Dense(26, activation="softmax"),    # Output layer
    ])

    # Compile the model
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Train the model
    model.fit(x_train, y_train, epochs=5, validation_data=(x_validate, y_validate))

    # plot the error and accuracy curves for the training data and validation datasets.
    plot_NNdata(model)
    return model


def NN_2(x_train, x_validate, y_train, y_validate):
    print("\n\n-----------------Neural Network 2-----------------")
    # Create the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),      # Flatten the input
        tf.keras.layers.Dense(16, activation="sigmoid"),    # Hidden layer
        tf.keras.layers.Dense(16, activation="sigmoid"),    # Hidden layer
        tf.keras.layers.Dense(26, activation="sigmoid"),    # Output layer
    ])

    # Compile the model
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Train the model
    model.fit(x_train, y_train, epochs=10, validation_data=(x_validate, y_validate))

    # plot the error and accuracy curves for the training data and validation datasets.
    plot_NNdata(model)
    return model


def plot_NNdata(model):
    plt.plot(model.history.history["loss"])
    plt.plot(model.history.history["val_loss"])
    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper left")
    plt.show()

    plt.plot(model.history.history["accuracy"])
    plt.plot(model.history.history["val_accuracy"])
    plt.title("Model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper left")
    plt.show()


def compare_models(model_1, model_2):
    print("\n\n-----------------Comparing Models-----------------")
    # Compare the models
    model_1_accuracy = model_1.history.history["val_accuracy"][-1]
    model_2_accuracy = model_2.history.history["val_accuracy"][-1]
    if model_1_accuracy > model_2_accuracy:
        print("Model 1 is better.")
        return model_1
    else:
        print("Model 2 is better.")
        return model_2


def test_best_model(best_model, x_test, y_test):
    print("\n\n-----------------Testing Best Model-----------------")
    # Predict
    y_pred = best_model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)

    print(f"f1_score: {f1_score(y_test, y_pred, average="weighted")}")
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.show()


if __name__ == "__main__":
    x_data, y_data = data_exploration()
    # train test split
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)
    # SVM
    # SVM_Linear(x_train, x_test, y_train, y_test)
    # SVM_NonLinear(x_train, x_test, y_train, y_test)

    # Split the training dataset into training and validation datasets.
    x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size=0.2)

    # Reshape the input to 28x28 image size
    x_train = x_train.values.reshape(-1, 28, 28)
    x_validate = x_validate.values.reshape(-1, 28, 28)
    x_test = x_test.values.reshape(-1, 28, 28)

    # Todo: Logistic Regression

    # NN
    model_1 = NN_1(x_train, x_validate, y_train, y_validate)
    model_2 = NN_2(x_train, x_validate, y_train, y_validate)

    # save the best model
    best_model = compare_models(model_1, model_2)
    best_model.save("best_model.keras")

    # load the best model
    best_model = tf.keras.models.load_model("best_model.keras")

    test_best_model(best_model, x_test, y_test)

    # TODO: Test the best model with images representing the alphabetical letters for the names of each member of your team.

    # TODO: Compare Logistic Regression, SVM, and Neural Network models and suggest the best model.

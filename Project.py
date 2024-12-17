import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
data_set = pd.read_csv("A_Z Handwritten Data.csv", header=None)  # header=None as the data has no header
print(data_set.describe())

# Separate features (pixel values) and labels (alphabets)
y = data_set.iloc[:, 0]   # All rows, only the first column (labels)
X = data_set.iloc[:, 1:]  # All rows, all columns except the first (784 pixels in row)

# Identify the number of unique classes
unique_classes = np.unique(y)

# Number of unique classes
print(f"Number of unique classes: {len(unique_classes)}")

# Show the distribution of the labels
sns.countplot(x=y)  # x-axis will be y (the labels)
plt.title("Distribution of The Labels")
plt.xlabel("Alphabet")
plt.ylabel("Frequency")
plt.show()

# Normalize the data
X = X / 255.0 

# To reshape
X = X.values.reshape(-1, 28, 28, 1) # -1 is the number of samples (372451), 28x28 is the size of the image, 1 is the number of channels (grayscale)

#================================================================================================
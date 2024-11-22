#%%
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

import os
from PIL import Image
import zipfile
#%%
data_path = "./reduced_wikiart/"                         # Just go to Justin's folder on the Google Drive and add a shortcut to "My_Drive"

# X = []
# y = []

split_X = []
split_y = []

for artistic_movement in os.listdir(data_path):
    temp_X = []
    temp_y = []
    movement_path = os.path.join(data_path, artistic_movement)                        # This returns the path for each art-movement folder (like "/content/drive/MyDrive/processed_wikiart/Abstract_Expressionism")
    print(movement_path)
    for file_name in os.listdir(movement_path):                                       # This indicates the image file name
        file_path = os.path.join(movement_path, file_name)                            # This returns the full path to the image
        img = Image.open(file_path).convert('L')                                      # Convert the image to grayscale
        img_array = np.array(img)                                                     # Convert the image to a numpy array
        #X.append(np.concatenate(img_array))                                           # Append the flattened image array (flattens each of the 256 image row vectors into one vector) to X
        #y.append(artistic_movement)                                                   # Append the artistic movement to y
        temp_X.append(np.concatenate(img_array))
        temp_y.append(artistic_movement)
    split_X.append(temp_X)
    split_y.append(temp_y)
#%%
len(split_X[0]), len(split_X[1]), len(split_X[2]), len(split_X[3]), len(split_X[4])
#%%
split_y[0][0], split_y[1][0], split_y[2][0], split_y[3][0], split_y[4][0]
#%%
equal_length = 6450

Romanticism_img = np.array(split_X[0][:equal_length])
Realism_img = np.array(split_X[1][:equal_length])
Post_Impressionism_img = np.array(split_X[2][:equal_length])
Impressionism_img = np.array(split_X[3][:equal_length])
Expressionism_img = np.array(split_X[4][:equal_length])

Romanticism_class = np.array(split_y[0][:equal_length])
Realism_class = np.array(split_y[1][:equal_length])
Post_Impressionism_class = np.array(split_y[2][:equal_length])
Impressionism_class = np.array(split_y[3][:equal_length])
Expressionism_class = np.array(split_y[4][:equal_length])
#%%
new_X = np.concatenate((Romanticism_img, Realism_img, Post_Impressionism_img, Impressionism_img, Expressionism_img))
new_y = np.concatenate((Romanticism_class, Realism_class, Post_Impressionism_class, Impressionism_class, Expressionism_class))
#%%
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(new_X, new_y, test_size = 0.2, random_state = 42, stratify = new_y)
#%%
# Create a non-linear kernel SVM classifier
svm = SVC(kernel = "rbf")
svm.fit(X_train, y_train)
#%%
predictions = svm.predict(X_test)
print(predictions)

# Evaluate the predictions
accuracy = svm.score(X_test, y_test)
print("Accuracy of SVM:", accuracy)
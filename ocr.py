import easyocr

reader = easyocr.Reader(['en']) # choose English as reading language

import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import os

# Parameters
image_directory = 'archive/processed_images'
img_height, img_width = 224, 224  # Should match your preprocessing size
batch_size = 32

# Load and preprocess images
def load_images(image_directory, target_size=(224, 224)):
    images = []
    labels = []  # Assuming labels can be derived from file names or another source

    for filename in os.listdir(image_directory):
        if filename.endswith('.png'):
            img_path = os.path.join(image_directory, filename)
            img = cv2.imread(img_path)
            img = cv2.resize(img, target_size)
            images.append(img)
            # Extract label from filename or another source
            label = reader.readtext(img)
            #label = filename.split('_')[0]  # Modify as per your naming convention
            labels.append(label)

    return np.array(images), np.array(labels)

images, labels = load_images(image_directory, (img_height, img_width))

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)



# display the plates with detected text
#plt.figure(figsize=(30,10))
#for i, plate in enumerate(plate_image_list):
#  plt.subplot(1, len(plate_image_list), i+1)
#  plt.imshow(cv2.cvtColor(plate, cv2.COLOR_BGR2RGB))

#  bounds = reader.readtext(plate) # read text
#  title_text = ''
#  for text in bounds:
#    title_text += text[1] + ' '
#  plt.title('Detected Plate Number: ' + title_text, fontdict={'fontsize':20})
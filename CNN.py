from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import os
import numpy as np
from PIL import Image


(x_train, y_train), (x_test, y_test) = mnist.load_data()


def save_images(images, labels, base_dir):
    for idx, (img, label) in enumerate(zip(images, labels)):
        label_dir = os.path.join(base_dir, str(label))
        os.makedirs(label_dir, exist_ok=True)
        img_path = os.path.join(label_dir, f"{idx}.png")
        Image.fromarray(img).save(img_path)

save_images(x_train, y_train, "mnist_images/train")
save_images(x_test, y_test, "mnist_images/test")


x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype("float32") / 255
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype("float32") / 255

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


model = Sequential([
    Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, kernel_size=(3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

epoch=100

history=model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epoch, batch_size=128)

import matplotlib.pyplot as plt
font = {'family': 'Times New Roman',
        'weight': 'bold',
        'size': 14}
plt.rc('font', **font)
plt.figure(figsize=(6,4))
plt.plot(range(epoch),history.history["accuracy"],label="Training Accuracy")
plt.plot(range(epoch),history.history["val_accuracy"],label="Validation Accuracy")
plt.xlabel("Number of Epochs",fontweight="bold",fontsize=18)
plt.ylabel("Accuracy",fontweight="bold",fontsize=18)
plt.xticks(fontweight="bold",fontsize=16)
plt.yticks(fontweight="bold",fontsize=16)
plt.xlim(0,100)
plt.legend()
plt.show()
plt.tight_layout()
plt.savefig('Result/Accuracy.jpg',dpi=800)

plt.figure(figsize=(6,4))
plt.plot(range(epoch),history.history["loss"],label="Training loss")
plt.plot(range(epoch),history.history["val_loss"],label="Validation loss")
plt.xlabel("Number of Epochs",fontweight="bold",fontsize=18)
plt.ylabel("loss",fontweight="bold",fontsize=18)
plt.xticks(fontweight="bold",fontsize=16)
plt.yticks(fontweight="bold",fontsize=16)
plt.xlim(0,100)
plt.legend()
plt.show()
plt.tight_layout()
plt.savefig('Result/loss.jpg',dpi=800)


model.save("Number_Identification_model.keras")
from tensorflow.keras.models import load_model
Model=load_model('Number_Identification_model.keras')

import matplotlib.pyplot as plt
import cv2
image = cv2.imread('mnist_images/train/7/1001.png')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
resized_img = cv2.resize(image_rgb, (28, 28)).astype(np.float32) / 255.0
one_image_batch = np.expand_dims(resized_img, axis=0) 
predicted = Model.predict(one_image_batch)
print("Predicted class:", np.argmax(predicted))
plt.figure()
plt.imshow(resized_img,cmap="gray")
plt.axis('off')
plt.show()
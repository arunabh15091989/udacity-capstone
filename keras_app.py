from prediction_scripts.detector_functions import face_detector, dog_detector
from prediction_scripts.data_functions import path_to_tensor
from keras.models import load_model
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import pickle
import numpy as np
from os import listdir
from os.path import isfile, join
import cv2
from flask import request
from flask import jsonify
from flask import flask
from keras.preprocessing import image
from tqdm import tqdm
from keras.applications import resnet50
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
import base64
import io
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array



app = Flask(__name__)

def get_inception_model():
    gloabl Inception_model
    Inception_model = Sequential()
    Inception_model.add(GlobalAveragePooling2D(input_shape=train_Inception.shape[1:]))
    Inception_model.add(Dense(500, activation='relu'))
    Inception_model.add(Dropout(0.4))
    Inception_model.add(Dense(133, activation='softmax'))
    Inception_model.load_weights('models/weights.best.Inception.hdf5')
    print('...Inception model Loaded')


def preprocess_image(image,target_size):
    if image.mode !="RGB":
        image=image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image,axis=0)
    return image



@app.route('/hello',methods=['POST'])
def make_prediction():
    message = request.get_json(force=True)
    encoded = message["image"]
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image, target_size=(224, 224))
    plt.axis('off')
    if dog_detector(image):
        print("Good Boy!You are a dog")
        #imgplot = plt.imshow(image)
        return jsonify("You are a {}".format(predict_breed(image)),image)

    if face_detector(image):
        print("Not a good boy! You are a Human")
        #imgplot = plt.imshow(image)
        return jsonify("Mmmmmm....If you were a dog, I guess you would be a ... {}!!".format(predict_breed(image)),image)

    else:
        return ("Hmmm... seems this neither dog, nor human; must be something else ."),image

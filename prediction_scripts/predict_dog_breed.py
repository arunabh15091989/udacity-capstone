# import packages
from prediction_scripts.detector_functions import face_detector, dog_detector
from prediction_scripts.data_functions import path_to_tensor
from keras.models import load_model
from keras.applications.resnet50 import ResNet50, preprocess_input
import pickle
import numpy as np
from os import listdir
from os.path import isfile, join
#from keras import backend as K
def make_prediction(img_path):
    img = mpimg.imread(img_path)
    plt.axis('off')
    if dog_detector(img_path):
        print("Good Boy!You are a dog")
        imgplot = plt.imshow(img)
        return "You are a {}".format(predict_breed(img_path)),img

    if face_detector(img_path):
        print("Not a good boy! You are a Human")
        imgplot = plt.imshow(img)
        return "Mmmmmm....If you were a dog, I guess you would be a ... {}!!".format(predict_breed(img_path)),img

    else:
        return ("Hmmm... seems this neither dog, nor human; must be something else ."),img

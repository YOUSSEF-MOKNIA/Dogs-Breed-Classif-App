import numpy as np
import tf_keras
import tensorflow_hub as hub
import tensorflow as tf

filename = "DogClassifModel.h5"
import os
model_pth = os.path.join(os.path.dirname(__file__), filename)

def load_dog_breed_model(model_path=model_pth):
    model = tf_keras.models.load_model(model_path, custom_objects={"KerasLayer":hub.KerasLayer})
    model.trainable = False
    return model

IMG_SIZE = 224

def preprocess_image(image_tensor):

    # Normalize the image values from 0-255 to 0-1
    image_tensor = image_tensor / 255.0

    # Resize the image to the desired size
    image_tensor = tf.image.resize(image_tensor, size=[IMG_SIZE, IMG_SIZE])
    
    # Add batch dimension
    image_tensor = tf.expand_dims(image_tensor, axis=0)
    
    return image_tensor

def predict_breed(model, image):
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return predicted_class

def get_breed_name(class_index):
    CLASS_NAMES = [
    'affenpinscher', 'afghan_hound', 'african_hunting_dog', 'airedale',
    'american_staffordshire_terrier', 'appenzeller', 'australian_terrier',
    'basenji', 'basset', 'beagle', 'bedlington_terrier', 'bernese_mountain_dog',
    'black-and-tan_coonhound', 'blenheim_spaniel', 'bloodhound', 'bluetick',
    'border_collie', 'border_terrier', 'borzoi', 'boston_bull',
    'bouvier_des_flandres', 'boxer', 'brabancon_griffon', 'briard',
    'brittany_spaniel', 'bull_mastiff', 'cairn', 'cardigan',
    'chesapeake_bay_retriever', 'chihuahua', 'chow', 'clumber', 'cocker_spaniel',
    'collie', 'curly-coated_retriever', 'dandie_dinmont', 'dhole', 'dingo',
    'doberman', 'english_foxhound', 'english_setter', 'english_springer',
    'entlebucher', 'eskimo_dog', 'flat-coated_retriever', 'french_bulldog',
    'german_shepherd', 'german_short-haired_pointer', 'giant_schnauzer',
    'golden_retriever', 'gordon_setter', 'great_dane', 'great_pyrenees',
    'greater_swiss_mountain_dog', 'groenendael', 'ibizan_hound', 'irish_setter',
    'irish_terrier', 'irish_water_spaniel', 'irish_wolfhound',
    'italian_greyhound', 'japanese_spaniel', 'keeshond', 'kelpie',
    'kerry_blue_terrier', 'komondor', 'kuvasz', 'labrador_retriever',
    'lakeland_terrier', 'leonberg', 'lhasa', 'malamute', 'malinois', 'maltese_dog',
    'mexican_hairless', 'miniature_pinscher', 'miniature_poodle',
    'miniature_schnauzer', 'newfoundland', 'norfolk_terrier',
    'norwegian_elkhound', 'norwich_terrier', 'old_english_sheepdog',
    'otterhound', 'papillon', 'pekinese', 'pembroke', 'pomeranian', 'pug',
    'redbone', 'rhodesian_ridgeback', 'rottweiler', 'saint_bernard', 'saluki',
    'samoyed', 'schipperke', 'scotch_terrier', 'scottish_deerhound',
    'sealyham_terrier', 'shetland_sheepdog', 'shih-tzu', 'siberian_husky',
    'silky_terrier', 'soft-coated_wheaten_terrier', 'staffordshire_bullterrier',
    'standard_poodle', 'standard_schnauzer', 'sussex_spaniel', 'tibetan_mastiff',
    'tibetan_terrier', 'toy_poodle', 'toy_terrier', 'vizsla', 'walker_hound',
    'weimaraner', 'welsh_springer_spaniel', 'west_highland_white_terrier',
    'whippet', 'wire-haired_fox_terrier', 'yorkshire_terrier'
]

    breed_name = CLASS_NAMES[class_index]
    
    return breed_name

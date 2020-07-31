import pandas as pd
import numpy as np

#tensorflow
import tensorflow
from tensorflow.keras.models import load_model
print('import successfull')

#loading model file 
HEALTH_MODEL_PATH = './models/health_classifier.h5'
SS_MODEL_PATH = './models/Subspecies_classifier.h5'

#loading model
health_model = load_model(HEALTH_MODEL_PATH)
ss_model= load_model(SS_MODEL_PATH)

health_cls={
    1:'Varroa, Small Hive Beetles',
    2:'ant problems',
    3: 'few varrao, hive beetles',
    4: 'healthy',
    5 :'hive being robbed',
    6:'missing queen'
}

ss_cls={
    1 :'-1',
    2: '1 Mixed local stock 2',
    3 :'Carniolan honey bee',
    4: 'Italian honey bee',
    5: 'Russian honey bee',
    6 :'VSH Italian honey bee',
    7: 'Western honey bee '
}

#making prediction 
def model_predict(file_path, model, cls):
    from tensorflow.keras.preprocessing import image
    # loading the images, target size should be equal to input size of image while training , we used earlier (100,100) on training 
    img = image.load_img(file_path, target_size=(100,100))

    # Preprocessing the image
    x = image.img_to_array(img)
    ## Scaling
    x=x/255
    x = np.expand_dims(x, axis=0)

    #image = np.expand_dims(image, axis=0)
    #image = np.array(image)
    pred = model.predict_classes([x])[0]
    sign = cls[pred+1]
    return sign


#images that needs to be predicted 
file_path= './Dataset/bee_imgs/041_067.png'
health_pred = model_predict(file_path, health_model,health_cls)
ss_pred =  model_predict(file_path,ss_model,ss_cls)
print(health_pred)
print(ss_pred)



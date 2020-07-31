
![GitHub repo size](https://img.shields.io/github/repo-size/Uttam580/Honey_Bees_Classifierir?style=plastic)
![GitHub language count](https://img.shields.io/github/languages/count/Uttam580/Honey_Bees_Classifier?style=plastic)
![GitHub top language](https://img.shields.io/github/languages/top/Uttam580/Honey_Bees_Classifier?style=plastic)
![GitHub last commit](https://img.shields.io/github/last-commit/Uttam580/Honey_Bees_Classifier?color=red&style=plastic)


Medium Blog: 

<a href ="https://medium.com/@uttam94/"> <img src="https://github.com/Uttam580/Uttam580/blob/master/img/medium.png" width=60 height=30>


# Honey Bees Classifier

## Objective: 

  In this project, the objective is to predict  strength and health of honey bees.
  frequent check-ups on the hive are time-consuming and disruptive to the bees' workflow and hive in general.
    By understanding the bees we can understand hive itself. 

    * How can we improve our understanding of a hive through images of bees?

    * How can we expedite the hive checkup process?

    * How can bee image data help us recognize problems earlier?

    * How can bee image data help us save our bees?


## Dataset :

  Dataset with adnotated images of bees from various locations of US, captured over several months during 2018, at different hours, from various bees subspecies, and with different health problems.
  Data has downloded form kaggle .Use Below link to download the dataset.

Dataset : <a href="https://www.kaggle.com/jenny18/honey-bee-annotated-images">Honey Bees Data </a>

  This dataset contains 5,100+ bee images annotated with location, date, time, subspecies, health condition, caste, and pollen


**quick demo**

  ![demo_gif](https://github.com/Uttam580/Honey_Bees_Classifier/blob/master/demo.gif)


## Technical Aspect

1. Training a deep learning model using tensorflow. I trained model on local system using NVIDIA GEFORCE GTX   1650 for for two models (subSpecies Classifier and health_classifier). I have to train 5k images for both the models. Both models trained for 30 epochs and on 32 batch size.

###### ```To check if training  is acelearted by gpu or not```

    import tensorflow as tf 

    from tensorflow.python.client import device_lib

    print(tf.test.is_built_with_cuda())

    print(device_lib.list_local_devices())

Below is the neural network architect of trained model.

**```Subspecies model```**            |  **```Health model```**
:-------------------------:|:-------------------------:
![Subspecies model](https://github.com/Uttam580/Honey_Bees_Classifier/blob/master/Subspecies_classifier.h5.png) |  ![Health model ](https://github.com/Uttam580/Honey_Bees_Classifier/blob/master/health_classifier.h5.png)


2. Building and hosting using FLASK.

## Directory Tree

```
honey_bees_classifier
├─ Dataset
├─ logger
├─ models
│  └─ eda
├─ static
│  ├─ css
│  ├─ images
│  └─ js
├─ subspecies_files
├─ templates
└─ uploads
```


##  Contents

*```Dataset``` :  Contains raw data for training (images , csv )

*```logger```  :  contains log file while training the model so that we can check the model perfomance  after training.

*```models```  :  contains model training script and trained model file

*```static```  :  static part of UI

*```templates```: frontend templates of UI

*```uploads```  :   when images is uploaded it will save in uploads and will use for prediction.

*```standalone.py``` : simple standalone py script for prediction. 


  
## Installation

* Clone this repository and unzip it.

* create new env with python 3 and activate it .

* Install the required packages using pip install -r requirements.txt

* Execute the command: python app.py

* Open ```http://127.0.0.1:5000/``` in your browser.

## Technologies Used

![](https://forthebadge.com/images/badges/made-with-python.svg)

[<img target="_blank" src="https://www.tensorflow.org/images/tf_logo_social.png" width=280>](https://www.tensorflow.org)[<img target="_blank" src="https://flask.palletsprojects.com/en/1.1.x/_images/flask-logo.png" width=170>](https://flask.palletsprojects.com/en/1.1.x/) 

## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

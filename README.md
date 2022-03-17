# Deep Learning practice image classification using Tensorflow &amp; Keras
- Created a tool that classify images of Tomato plant using tensorflow CNN model (Test Accuracy ~ 90%).
- Downloaded the data set from Kaggle : https://www.kaggle.com/arjuntejaswi/plant-village
	- Note: The dataset consists of ~16000 images under 10 categories. The current model is built on a subset of the data (~3000 images) due to hardware constraints.
- After resizing, rescaling and data augmentation of the images, the CNN model (6 convolution and max pooling layers) is built.
- Built a client facing API using fastAPI
## Code and Resources Used
Python Version: 3.7\
Packages: pandas, numpy, tensorflow, matplotlib, fastapi, uvicorn, io, PIL\
For Web Framework Requirements: ```pip install -r requirements.txt```\
Dataset: https://www.kaggle.com/arjuntejaswi/plant-village

## YouTube Project Walk-Through
Thanks to Dhaval for his videos on Image classification and walk-through of the end to end project
https://www.youtube.com/playlist?list=PLeo1K3hjS3ut49PskOfLnE6WUoOp_2lsD

## How To:
- Create conda env (optional): Open command prompt and go to required directory and enter command ```conda create --prefix ./env_name python=3.7```
  - Activate conda env: ```conda activate env_name```
- Clone git repo (git bash): ```git init``` ```git clone repo_url```
- In conda env (command prompt):
  - Install packages : ```pip install -r requirements.txt```
- To test API:
  - In conda env (command prompt): go to the path /api/ ```python main.py```
  - Go to browser and enter url http://localhost:8000/docs
	- Choose 'Predict' (post method) and click on Tryout and browse the image to test and click on execute.

## Model Building

- First, splitted the data into train, validation and test datasets of 80%, 10% and 10% respectively.

- Defined model layers as below
	- Resizing: (256,256)
	- Rescaling: (1.0/255)
	- RandomFlip: horizontal_and_vertical
	- RandomRotation: 0.2
	- Conv2D: (filters:32 or 64, kernel_size = (3,3), activation = 'relu'
	- MaxPooling2D: pool_size=(2, 2)
	- Flatten
	- Dense: units = 64, activation = 'relu'
	- Dense: units = 10, activation = 'softmax'
	
### Notes:
	- /models/params.py: Contains all the parameters that are used to built model
	- /models/commons.py: Contains commonly used functions
	- /models/tf_cnn_model.py: Contains code for building model using tensorflow.
	- /api/main.py: Contains code for API

## Model performance
The model accuracy and loss after 50 epochs is below.\
![Alt text](/models/Training_validation_accuracy_loss.png?raw=true "Training_Validation Accuracy and Loss") \
Prediction of Random Tomato Leaves.\
![Alt text](/models/predicted_12_tomato_leaves.png?raw=true "Prediction of Random Tomato Leaves")
Prediction using FastAPI.\
![Alt text](/models/output.PNG?raw=true "Prediction using FastAPI")


## Productionization
In this step, I built a FastAPI endpoint that was hosted on a local webserver by following along with the youtube tutorial in the reference section above. The API endpoint takes in a request with image as input and returns predicted class and confidence values.

Reference: 
- https://www.tensorflow.org

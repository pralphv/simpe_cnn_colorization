# simple_cnn_colorization

Dependencies:
- keras
- numpy
- PIL
- skimage

This project is a simple colorization program using Keras. 
CNN is used, with a Unet structure. 
Loss function is Mean Absolute Error.

1. Put training images to folder "train". 
2. The program will check if folder "batches" is empty. 
3. If folder "batches" is empty, it will take images from folder "trian" and turn them into numpy arrays and save them in folder "batches" (make sure each file is smaller than your available RAM)
4. The program will check if folder "checkpoint" is empty.
5. If folder "checkpoint" is empty, it will create a new model
6. Training starts.
7. If you want to see your progress, set predict_while_training = True in main.py and put images into folder "predict". Predictions will be outputed in folder "result"


Example 1:

![alt text](https://github.com/pralphv/simple_cnn_colorization/blob/master/photos/predict_1.jpg)   ![alt text](https://github.com/pralphv/simple_cnn_colorization/blob/master/photos/result_1.jpeg)

Example 2:

![alt text](https://github.com/pralphv/simple_cnn_colorization/blob/master/photos/predict_2.jpg)   ![alt text](https://github.com/pralphv/simple_cnn_colorization/blob/master/photos/result_2.jpeg)

Example 3:

![alt text](https://github.com/pralphv/simple_cnn_colorization/blob/master/photos/predict_3.jpg)   ![alt text](https://github.com/pralphv/simple_cnn_colorization/blob/master/photos/result_3.jpeg)

import os
import numpy as np
from PIL import Image
from skimage.transform import resize
from skimage.io import imsave
from skimage.color import rgb2lab, lab2rgb


def save_numpy(no_of_parts):

    def load_image(file):
        image = Image.open(file)
        image = np.array(image, dtype=np.float32) / 255  # if float, need to be within 0-1. int within 0-255
        return image

    path = 'train\\'
    files = os.listdir(path)
    no_of_files = len(files)
    images = []
    for file in files:
        image = load_image(file=path+file)
        images.append(image)

    images = np.array(images, dtype=np.float32)
    np.random.shuffle(images)

    start = 0
    end = 0
    for i in range(1,no_of_parts+1):
        end = i/no_of_parts
        np.save('batches\\kimi_images_{}'.format(i), images[int(no_of_files*start):int(no_of_files*end)])
        start = end

def image_to_lab(image):
    image_lab = rgb2lab(image)
    L = image_lab[:, :, 0]
    A = image_lab[:, :, 1]
    B = image_lab[:, :, 2]
    return L,A,B

def load_image(file):
    image= Image.open(file)
    image = np.array(image)
    return image

def process_image(file):
    image = load_image(file=file)
    image = np.array(image,dtype=np.float32)/255 #if float, need to be within 0-1. int within 0-255
    if len(image.shape) != 3:
        return None, None

    L, A, B = image_to_lab(image)

    L = resize(L, (256, 256), mode='constant')
    A = resize(A, (256, 256), mode='constant')
    B = resize(B, (256, 256), mode='constant')

    L = L/100
    A = A/128
    B = B/128

    x = L
    y = np.dstack((A, B))

    return x,y

def predict_image(model,file):
    x_temp, y = process_image(file=file)
    x = x_temp[np.newaxis, :, :, np.newaxis]
    output = model.predict(x)[0]
    cur = np.zeros((256, 256, 3))
    cur[:,:,0] = x_temp*100
    cur[:,:,1] = output[:,:,0]*128
    cur[:,:,2] = output[:,:,1]*128

    existing_files = os.listdir('result')
    number = 1

    while 'image_'+str(number)+'.jpeg' in existing_files:
        number += 1

    imsave("result\\image_"+str(number)+".jpeg", lab2rgb(cur))

from keras.preprocessing.image import ImageDataGenerator
import models
from keras.models import load_model
from datetime import datetime
from image_utility import *

def augment_generator(images):
    batches = []
    datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=20,
        horizontal_flip=True,
        vertical_flip=True)

    for batch in datagen.flow(images, batch_size=1):
        batches.append(batch[0])
        if len(batches) == images.shape[0]:
            break
    batches = np.array(batches)
    return batches

def dateNow():
    dateName = str(datetime.today())
    dateName = dateName.replace('-','')
    dateName = dateName.replace(' ','')
    dateName = dateName.replace(':','')
    dateName = dateName[:13]
    return dateName

def predict(model):
    for predict_file in os.listdir('predict'):
        predict_image(model=model, file='predict\\'+ predict_file)

def check_newest():
    files = os.listdir('checkpoint')
    newest = int(files[0][6:-3])
    for file in files:
        file = int(file[6:-3])
        if file > newest:
            newest = file
    newest = 'model_{}.h5'.format(newest)
    return newest

def train(model,batch_size,predict_while_training):
    batch_start = 0
    batch_end = batch_start + batch_size

    steps = 0

    batches = os.listdir('batches')
    while True:
        for file in batches:
            file = 'batches\\'+file
            images = np.load(file)
            images = augment_generator(images)
            x = []
            y = []
            for image in images:
                L,A,B = image_to_lab(image)
                x.append(L)
                temp = np.dstack((A, B))
                y.append(temp)
            x = np.array(x,dtype=np.float32)/100
            x = x[:, :, :, np.newaxis]
            y = np.array(y,dtype=np.float32)/128

            no_of_files = y.shape[0]
            while batch_start < no_of_files:
                print("Batch: ", int(batch_start/no_of_files*100), '%')
                batch_x = x[batch_start:batch_end]
                batch_y = y[batch_start:batch_end]
                batch_start += batch_size
                batch_end += batch_size
                model.fit(x = batch_x,
                          y = batch_y,
                          epochs = 1
                          )
                if steps % 1000 == 0 :
                    model.save('checkpoint\\model_'+dateNow()+'.h5')
                    if predict_while_training == True:
                        predict(model)
                steps += 1
            batch_start = 0
            batch_end = batch_size

def main():
    no_of_parts = 5 #for saving "train\" folder images to numpy
    batch_size = 1 #batch normalization is used. 1 is best
    predict_while_training = True

    if len(os.listdir('batches')) == 0:
        print("Found no batches. Saving Numpy from folder 'train\\'")
        save_numpy(no_of_parts)

    if len(os.listdir('checkpoint')) == 0:
        print("Found no save points. Creating new model")
        model = models.get_unet(img_rows=256,
                                img_cols=256,
                                dimensions=1)
    else:
        newest = check_newest()
        print("Found Save Point. Using {}".format(newest))
        model = load_model('checkpoint\\{}'.format(newest))
    train(model, batch_size, predict_while_training)


if __name__ == '__main__':
    main()
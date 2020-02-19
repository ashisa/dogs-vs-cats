from __future__ import print_function
import numpy as np
import sys
import os
import argparse


###################################################################
# Variables #
# When launching project or scripts from Visual Studio, #
# input_dir and output_dir are passed as arguments.  #
# Users could set them from the project setting page.  #
###################################################################
input_dir = '.'
output_dir = 'output'
log_dir = 'logs'
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
img_width, img_height = 150, 150
batch_size = 32
epochs = 1
train_samples = 2048
validation_samples = 832
desired_size = 150

#################################################################################
# Keras configs.  #
# Please refer to https://keras.io/backend .  #
#################################################################################
import keras
from keras import backend as K

#K.set_floatx('float32')
#String: 'float16', 'float32', or 'float64'.

#K.set_epsilon(1e-05)
#float.  Sets the value of the fuzz factor used in numeric expressions.

#K.set_image_data_format('channels_first')
#data_format: string.  'channels_first' or 'channels_last'.


#################################################################################
# Keras imports.  #
#################################################################################
from keras import applications
from keras.models import Model
from keras.models import load_model
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import Layer
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Convolution2D
from keras.layers import ZeroPadding2D
from keras import optimizers
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from PIL import Image

datagen = ImageDataGenerator(rescale=1. / 255)

def keras_augment():
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    image = load_img('data/train/dogs/dog.0.jpg')
    x = img_to_array(image)  
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

    # the .flow() command below generates batches of randomly transformed images
    # and saves the results to the `preview/` directory
    i = 0
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir='preview', save_prefix='dog', save_format='jpeg'):
        i += 1
        if i > 20:
            break  # otherwise the generator would loop indefinitely

def small_cnn():
    # a simple stack of 3 convolution layers with a ReLU activation and
    # followed by max-pooling layers.
    model = Sequential()
    model.add(Convolution2D(32, (3, 3), input_shape=(img_width, img_height,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='SGD',
                  metrics=['accuracy'])

    return model

def save_model(model, name):
    # serialize model to JSON
    model_json = model.to_json()
    with open("output/" + name + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("output/" + name + ".h5")

def train_smallcnn():
    # automagically retrieve images and their classes for train and validation
    # sets
    train_generator = datagen.flow_from_directory(train_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='binary')

    validation_generator = datagen.flow_from_directory(validation_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='binary')

    # get model
    model = small_cnn()

    # train this model on the dataset
    model.fit_generator(train_generator,
            steps_per_epoch=train_samples // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validation_samples // batch_size, verbose=1)

    # save the model weights
    save_model(model, "basic_cnn")
    return model, validation_generator, validation_samples

def evaluate(model, validation_generator, validation_samples):
    result = model.evaluate_generator(validation_generator, validation_samples)
    return result

def train_augmented_data():
    # Augment images and their classes for train and validation sets
    train_datagen = ImageDataGenerator(rescale=1. / 255,        # normalize pixel values to [0,1]
            shear_range=0.2,       # randomly applies shearing transformation
            zoom_range=0.2,        # randomly applies zoom transformation
            horizontal_flip=True)  # randomly flip the images

    # same code as before
    train_generator = train_datagen.flow_from_directory(train_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='binary')

    validation_generator = datagen.flow_from_directory(validation_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='binary')

    # get model
    model = small_cnn()

    # Training the model with augmented data
    model.fit_generator(train_generator,
            steps_per_epoch=train_samples // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validation_samples // batch_size,)

    # save the model weights
    save_model(model, 'basic_cnn_augmented')
    return model, validation_generator, validation_samples

def use_pretrained_model():
    # Using a pretrained model
    # Loading the model
    model_vgg = applications.VGG16(include_top=False, weights='imagenet', input_shape=(150, 150, 3))

    train_generator = datagen.flow_from_directory(train_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode=None,
            shuffle=False)

    validation_generator = datagen.flow_from_directory(validation_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode=None,
            shuffle=False)

    # Save the features weights
    features_train = model_vgg.predict_generator(train_generator, train_samples // batch_size)
    np.save(open('output/features_train.npy', 'wb'), features_train)

    # Save the validation weights
    features_validation = model_vgg.predict_generator(validation_generator, validation_samples // batch_size)
    np.save(open('output/features_validation.npy', 'wb'), features_validation)

    # Load the weights
    train_data = np.load(open('output/features_train.npy', 'rb'))
    train_labels = np.array([0] * (train_samples // 2) + [1] * (train_samples // 2))

    validation_data = np.load(open('output/features_validation.npy', 'rb'))
    validation_labels = np.array([0] * (validation_samples // 2) + [1] * (validation_samples // 2))

    # Define the fully connected network
    model_top = Sequential()
    model_top.add(Flatten(input_shape=train_data.shape[1:]))
    model_top.add(Dense(256, activation='relu'))
    model_top.add(Dropout(0.5))
    model_top.add(Dense(1, activation='sigmoid'))

    model_top.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model now
    model_top.fit(train_data, train_labels,
            epochs=epochs, 
            batch_size=batch_size,
            validation_data=(validation_data, validation_labels))

    # Train the model now
    model_top.fit(train_data, train_labels,
            epochs=epochs, 
            batch_size=batch_size,
            validation_data=(validation_data, validation_labels))

    # Save the weights from this model
    save_model(model_top, 'bottleneck')
    return model_top, validation_data, validation_labels

def finetune_model():
    # Loading the model
    model_vgg = applications.VGG16(include_top=False, weights='imagenet', input_shape=(150, 150, 3))

    # Start with a fully trained-classifer
    top_model = Sequential()
    top_model.add(Flatten(input_shape=model_vgg.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1, activation='sigmoid'))

    # Use the weights from the earlier model
    top_model.load_weights('output/bottleneck.h5')

    # Add this model on top of the convolutional base.
    model = Model(inputs = model_vgg.input, outputs = top_model(model_vgg.output))

    # Fine-tuning needs training only a few layers.  Set first 25 layers as
    # non-trainable
    for layer in model.layers[:15]:
        layer.trainable = False

    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])

    # Augment data and
    train_datagen = ImageDataGenerator(rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(train_data_dir,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(validation_data_dir,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='binary')

    # fine-tune the model
    model.fit_generator(train_generator,
        steps_per_epoch=train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_samples // batch_size)

    save_model(model, 'finetuned_model')
    return model, validation_generator, validation_samples

def load_image(img_path, show=False):

    img = image.load_img(img_path, target_size=(150, 150))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img_tensor[0])                           
        plt.axis('off')
        plt.show()

    return img_tensor

def predict_image():
    # loading model
    json_file = open('output/finetuned_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("output/finetuned_model.h5")

    # URL
    #sample_file_url = "https://shop.epictv.com/sites/default/files/ae42ad29e70ba8ce6b67d3bdb6ab5c6e.jpeg"
    #fd = urllib.urlopen(sample_file_url)
    #raw_data = io.BytesIO(fd.read())
    #image = Image.open(raw_data)

    for root, dirs, files in os.walk('modelfiles/test_images'):
        for name in files:
            image_path = os.path.join(root, name)
            image = Image.open(image_path)
            image = image.convert('RGB')
            old_size = image.size
            #Taking the max from height and width of the image and calculating ratio
            ratio = float(desired_size)/max(old_size)
            new_size = tuple([int(x*ratio) for x in old_size])

            image = image.resize(new_size, Image.ANTIALIAS)
            new_im = Image.new("RGB", (desired_size, desired_size), (255,255,255)) #creating RGB image and applying white mask for blank area
            new_im.paste(image, ((desired_size-new_size[0])//2, (desired_size-new_size[1])//2))
        
            x = np.asarray(new_im, dtype='float32')
            #x = x.transpose(2, 0, 1)
            x = np.expand_dims(x, axis=0)
            out = model.predict(x)
            print(name, out)

def main():
    ### Augmenting data
    keras_augment()

    ### 1 - build a small cnn model
    print('training simple cnn...')
    model, validation_generator, validation_samples = train_smallcnn()
    print('training completed.')
    
    print('evaluating model...')
    eval = evaluate(model, validation_generator, validation_samples)
    print(eval)

    ### 2 - training with augmented data, useful for increasing accuracy
    #print('training with augmented data...')
    #model, validation_generator, validation_samples = train_augmented_data()

    #print('evaluating model...')
    #eval = evaluate(model, validation_generator, validation_samples)
    #print(eval)

    ### 3 - training a pretrained model
    #print('training a pretrained model...')
    #model, validation_data, validation_labels = use_pretrained_model()

    #print('evaluating the model...')
    #eval = model.evaluate(validation_data, validation_labels)
    #print(eval)

    ### 4 - finetuning model
    #print('finetuning model...')
    #model, validation_generator, validation_samples = finetune_model()

    #print('evaluating the model...')
    #eval = evaluate(model, validation_generator, validation_samples)
    #print(eval)

    ## 5 - using model to predict category
    print('predicting images...')
    predict_image()

    #input('press enter to continue...')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, 
                        default=None, 
                        help="Input directory where where training dataset and meta data are saved", 
                        required=False)
    parser.add_argument("--output_dir", type=str, 
                        default=None, 
                        help="Input directory where where logs and models are saved", 
                        required=False)

    args, unknown = parser.parse_known_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    log_dir = output_dir

    main()

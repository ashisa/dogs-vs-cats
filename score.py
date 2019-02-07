from __future__ import print_function
import numpy as np
import json
import sys
import os
from keras.models import model_from_json
from keras.preprocessing import image
from PIL import Image

from azureml.core.model import Model


def load_image(img_path, show=False):
    img = image.load_img(img_path, target_size=(150, 150))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension
    img_tensor /= 255.                                      # imshow expects values in the range[0,1]

    if show:
        plt.imshow(img_tensor[0])                           
        plt.axis('off')
        plt.show()

    return img_tensor

def init():
    print('init() called...')
    global model
    for root, dirs, files in os.walk('.', topdown=False):
        for name in dirs:
            if name.endswith('modelfiles'):
                model_json_path = os.path.join(root, name, 'finetuned_model.json')
                model_weights_path = os.path.join(root, name, 'finetuned_model.h5')
                # loading model
                json_file = open(model_json_path, 'r')
                loaded_model_json = json_file.read()
                json_file.close()
                model = model_from_json(loaded_model_json)
                # load weights into new model
                model.load_weights(model_weights_path)
    print('init() finished...')

def run(raw_data):
    global out
    desired_size = 150
    print('run() called...')
    for root, dirs, files in os.walk('.'):
        for name in dirs:
            if name.endswith('test_images'):
                print(os.path.join(root, name))
                for root, dirs, files in os.walk(os.path.join(root, name)):
                    for name in files:
                        image_path = os.path.join(root, name)
                        image = Image.open(image_path)
                        image = image.convert('RGB')
                        old_size = image.size
                        #Taking the max from height and width of the image and calculating
                        #ratio
                        ratio = float(desired_size) / max(old_size)
                        new_size = tuple([int(x * ratio) for x in old_size])

                        image = image.resize(new_size, Image.ANTIALIAS)
                        new_im = Image.new("RGB", (desired_size, desired_size), (255,255,255)) #creating RGB image and applying white mask for blank area
                        new_im.paste(image, ((desired_size - new_size[0]) // 2, (desired_size - new_size[1]) // 2))
        
                        x = np.asarray(new_im, dtype='float32')
                        #x = x.transpose(2, 0, 1)
                        x = np.expand_dims(x, axis=0)
                        out = model.predict(x)
                        print(name, out)

    return json.dumps({'score': np.float64(out)}, sort_keys=True, indent=4, separators=(',', ': '))

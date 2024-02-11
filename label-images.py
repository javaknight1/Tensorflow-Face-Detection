import tensorflow as tf
import json
import numpy as np
from matplotlib import pyplot as plt
import os

def load_image(x): 
    byte_img = tf.io.read_file(x)
    img = tf.io.decode_jpeg(byte_img)
    return img

def main():
    for folder in ['train','test','val']:
        for file in os.listdir(os.path.join('data', folder, 'images')):
            
            filename = file.split('.')[0]+'.json'
            existing_filepath = os.path.join('data','labels', filename)
            if os.path.exists(existing_filepath): 
                new_filepath = os.path.join('data',folder,'labels',filename)
                os.replace(existing_filepath, new_filepath)      


if __name__ == "__main__":
    main()
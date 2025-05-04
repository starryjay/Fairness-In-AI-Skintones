import numpy as np
import pandas as pd 
import os
import shutil
import tensorflow as tf

def create_file_structure():
    '''file structure'''
    source = 'data/lfw-deepfunneled/'
    dest = 'data/'
    img = []
    os.makedirs(dest, exist_ok=True)
    for root, dirs, files in os.walk(source):
        for file in files:
            if file == '.DS_Store' or "DS_Store" in file:
                print("Skipping DS Store")
                continue
            person = os.path.basename(root)
            new_file = person + '_' + file
            src = os.path.join(root, file)
            dst = os.path.join(dest, new_file)
            shutil.copyfile(src, dst)
            img.append([person, new_file])
    df = pd.DataFrame(img, columns=['person', 'file'])
    df.to_csv('data/lfw.csv', index=False)

def preprocess_lfw(lfw_csv):
    '''preprocess lfw csv'''
    df = pd.read_csv(lfw_csv)
    one_per_person = df.groupby('person').sample(n=1, random_state=42)
    img_path = one_per_person['file']
    img_path = img_path.apply(lambda x: os.path.join('data', x))
    output_dir = 'data/resized/'
    os.makedirs(output_dir, exist_ok=True)
    for img in img_path:
        img = tf.io.read_file(img)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [128, 128])
        img = img / 255.0
        img_uint8 = tf.image.convert_image_dtype(img, tf.uint8)
        encoded = tf.image.encode_jpeg(img_uint8)
        tf.io.write_file(os.path.join(output_dir, os.path.basename(img)), encoded)

def main():
    create_file_structure()
    preprocess_lfw('data/lfw.csv')

main()
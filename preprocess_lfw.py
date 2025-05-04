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
            #remove DS store
            #print("Processing: ", file)
            if file == '.DS_Store' or "DS_Store" in file:
                print("Skipping DS Store")
                continue
            #get person 
            person = os.path.basename(root)
            #rename 
            new_file = person + '_' + file
            src = os.path.join(root, file)
            dst = os.path.join(dest, new_file)
            shutil.copyfile(src, dst)
            img.append([person, new_file])
    #df 
    df = pd.DataFrame(img, columns=['person', 'file'])
    df.to_csv('data/lfw.csv', index=False)

def preprocess_lfw(lfw_csv):
    '''preprocess lfw csv'''
    df = pd.read_csv(lfw_csv)
    #modify df to have one img per person randomly
    one_per_person = df.groupby('person').sample(n=1, random_state=42)
    # convert to 4d tensor [batch, height, width, channels]
    # get image path
    img_path = one_per_person['file']
    img_path = img_path.apply(lambda x: os.path.join('data', x))
    output_dir = 'data/resized/'
    os.makedirs(output_dir, exist_ok=True)
    for img in img_path:
        #read image
        img = tf.io.read_file(img)
        #decode image
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [128, 128])
        #normalize image
        img = img / 255.0
        img_uint8 = tf.image.convert_image_dtype(img, tf.uint8)
        encoded = tf.image.encode_jpeg(img_uint8)
        tf.io.write_file(os.path.join(output_dir, os.path.basename(img)), encoded)

def main():
    create_file_structure()
    preprocess_lfw('data/lfw.csv')

main()
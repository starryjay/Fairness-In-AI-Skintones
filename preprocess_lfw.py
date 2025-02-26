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
    resized = pd.DataFrame(columns=['img'])
    for img in img_path:
        #read image
        img = tf.io.read_file(img)
        #decode image
        img = tf.image.decode_jpeg(img, channels=3)
        #resize image to 224x224
        img = tf.image.resize(img, [128, 128])
        #normalize image
        img = img / 255.0
        #add to df
        resized = pd.concat([resized, pd.DataFrame({'img': [img]})], ignore_index=True)
    return resized

def main():
    #create_file_structure()
    resized = preprocess_lfw('data/lfw.csv')

main()
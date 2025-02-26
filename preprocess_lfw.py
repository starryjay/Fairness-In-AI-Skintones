import numpy as np
import pandas as pd 
import os
import shutil


def create_file_structure():
    '''file structure'''
    source = 'data/lfw-deepfunneled'
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
    #get unique persons
    persons = df['person'].unique()
    print(len(df))
    print("Number of persons: ", len(persons))

    #modify df to have one img per person randomly
    one_per_person = df.groupby('person').sample(n=1, random_state=42)
    print("One image per person", len(one_per_person))


def main():
    #create_file_structure()
    preprocess_lfw('data/lfw.csv')

main()
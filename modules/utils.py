# File containing all the necessary functions
# File: utils.py
# Author: Atharva Kulkarni


import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split





def read_data(path, delimiter=","):
    """ Function to read a csv/ tsv file.
    @param path (string): file path.
    @param columns (list): list of columns to read from the file.
    @delimiter (str): specify the delimiter by which the data is seperated. delimiter="," for csv and delimiter="\t" for tsv
    @return data (pd.DataFrame): DataFrame of the data.
    """
    return pd.read_csv(path, delimiter=delimiter, encoding="utf-8")





def clean_data(data):
    """ Function to clean the data (remove special characters, digits, punctuations, and extra spaces)
    @data (string): input text data to be cleaned.
    @ return data (string): cleaned text data.
    """
    data = "".join(re.sub('\W|\d+', ' ', data.lower()))
    data = re.sub('\s+', ' ', data)
    return data
    
    



def split_data(data, label, stratify=True, test_split=0.2, validation_split=0.2):
    """ Function to split the dataset into train-validation-test.
    @param data (pd.DataFrame): Data to be split.
    @param validation_split (float): Amount of validation data.
    @param text_split (float): Amount of test data.
    return training_data (pd.DataFrame): The training dataset.
    return validation_data (pd.DataFrame): The valiation dataset.
    return test_data (pd.DataFrame): The test dataset.
    """
    if stratify:
        training_data, test_data = train_test_split(data, test_size=test_split, stratify=data[label].values.tolist())
        training_data, validation_data = train_test_split(training_data, test_size=validation_split, stratify=training_data[label].values.tolist())
    else:
        training_data, test_data = train_test_split(data, test_size=test_split)
        training_data, validation_data = train_test_split(training_data, test_size=validation_split)
    return training_data, validation_data, test_data
    
    
    
    

def write_to_file(path, data):
    data.to_csv(path, sep="\t", index=False)




       
def get_embedding_dict(path):
    """ Function to read word embeddings into a dictionary.
    @param path (string): word embeddings file path.
    @return embedding_dict (dict): word embeddings dictionary.
    """
    embedding_dict = dict()
    with open(path, "r") as f:
        data = f.readlines()
    f.close()
    for row in data:
        row = row.split()
        if str(row[0]) == "nan":
            continue
        else:
            embedding_dict[row[0]] = np.array(row[1:], dtype="float32")
    return embedding_dict
    
    
    
    
    
def get_embedding_matrix(path):
    """ Function to read word embeddings into an array.
    @param path (string): word embeddings file path.
    @return embedding_matrix (np.array): word embeddings matrix.
    """
    embeddings = []
    with open(path, "r") as f:
        data = f.readlines()
    f.close()
    for row in data:
        row = row.split()
        if str(row[0]) == "nan":
            continue
        else:
            vec = [float(x) for x in row[1:]]
            embeddings.append(vec)
    embedding_matrix = np.array(embeddings , dtype="float32")
    return embedding_matrix
    

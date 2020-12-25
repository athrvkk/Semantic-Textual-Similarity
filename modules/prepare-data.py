# Python file to clean data and split it into train-validation-test datasets.
# File: prepare-data.py
# Author: Atharva Kulkarni

from utils import *
import argparse



if __name__ == '__main__':
    """ The main method to receive user inputs"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", default="../dataset/quora_duplicate_questions.tsv")
    parser.add_argument("--columns", default="question1,question2")
    parser.add_argument("--test_split", default="0.05")
    parser.add_argument("--validation_split", default="0.1")
    args = parser.parse_args()
           
    data = read_data(str(args.file_path), delimiter="\t")
    print("\nRead data!")
    
    columns = str(args.columns).split(",")
    for column in columns:
        data = data[data[column].notnull()]
    for column in columns:
        data[column] = data[column].apply(lambda record: clean_data(record))
    print("\nCleaned Data!")
    
    training_data, validation_data, test_data = split_data(data, label="is_duplicate", stratify=True, test_split=float(args.test_split), validation_split=float(args.validation_split))

    path = "/".join(str(args.file_path).split("/")[:-1])
    write_to_file(path+"/train.tsv", training_data)
    write_to_file(path+"/validation.tsv", validation_data)
    write_to_file(path+"/test.tsv", test_data)
    print("\nTrain Validation Test dataset created!\n")
    
        

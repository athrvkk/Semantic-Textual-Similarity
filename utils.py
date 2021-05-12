# File of the class Utils, containing various helper functions
# File: utils.py
# Author: Atharva Kulkarni


import pandas as pd
import numpy as np
import csv
import re
import spacy
import pandas as pd 
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import unicodedata

nltk.download('stopwords')
nltk.download('punkt')

from gensim.models import KeyedVectors
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences



class Utils():


    # -------------------------------------------- constructor --------------------------------------------
    
    def __init__(self):
        self.stop_words = stopwords.words('english')
        self.wordnet_lemmatizer = WordNetLemmatizer()
        self.nouns = ['NNP', 'NNPS']
        self.nlp = spacy.load('en_core_web_sm')
        self.tokenizer = Tokenizer(oov_token='[oov]')


    # -------------------------------------------- Function to read data --------------------------------------------
    
    def read_data(self, path):
        return pd.read_csv(path, 
                           header=None, 
                           sep="\t",  
                           usecols=[0, 1, 2, 4, 5, 6],
                           names=['genre', 'filename', 'year', 'score', 'sentence1', 'sentence2'],
                           quoting=csv.QUOTE_NONE, 
                           encoding='utf-8')
        



    # -------------------------------------------- Function to clean text --------------------------------------------
    
    def clean_text(self, text, remove_stopwords=True, lemmatize=True):
            """ Function to clean text
            @param text (str): text to be cleaned
            @param remove_stopwords (bool): To remove stopwords or not.
            @param lemmatize (bool): to lemmatize or not.
            """

            # Remove emails 
            text = re.sub('\S*@\S*\s?', '', text)
            
            # Remove new line characters 
            text = re.sub('\s+', ' ', text) 
            
            # Remove distracting single quotes 
            text = re.sub("\'", '', text)

            # Remove puntuations and numbers
            text = re.sub('[^a-zA-Z]', ' ', text)

            # Remove single characters
            text = re.sub('\s+[a-zA-Z]\s+^I', ' ', text)
            
            # Remove accented words
            text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

            # remove multiple spaces
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'^\s*|\s\s*', ' ', text).strip()
            text = text.lower()

            if not remove_stopwords and not lemmatize:
                return text

            # Remove unncecessay stopwords
            if remove_stopwords:
                text = word_tokenize(text)
                text = " ".join([word for word in text if word not in self.stop_words])
            
            # Word lemmatization
            if lemmatize:
                text = self.nlp(text)
                lemmatized_text = []
                for word in text:
                    if word.lemma_.isalpha():
                        if word.lemma_ != '-PRON-':
                            lemmatized_text.append(word.lemma_.lower())
                text = " ".join([word.lower() for word in lemmatized_text])
                    
            return text
        
        
        
        
        
    # -------------------------------------------- Function read dictionary --------------------------------------------
    
    def get_dict(self, path, key_column, value_column):
        """ Function to read a file into dictionary.
        @param path (str): path to file.
        return dict: created dictionary.
        """
        data = pd.read_csv(path)
        return dict(zip(data[key_column].values.tolist(), data[value_column].values.tolist()))



       
    # -------------------------------------------- Function to tokenize and pad input text --------------------------------------------    

    def tokenize_and_pad(self, df, maxlen=50, padding_type='post', truncating_type='post', mode="train"):
        """ Function to prepare text for model input (tokenize and pad).
        @param corpus (list): the corpus to prepare.
        @param maxlen (int): max allowed length of input texts.
        @param padding_type (str): padding type (post or pre).
        @param truncating_type (str): truncating type (post or pre).
        @mode (str): specify train or test mode.
        """
        sentence1 = df['sentence1'].apply(lambda x : self.clean_text(x, remove_stopwords=False, lemmatize=False)).values.tolist()
        sentence2 = df['sentence2'].apply(lambda x : self.clean_text(x, remove_stopwords=False, lemmatize=False)).values.tolist()

        if mode == "train":
            corpus = sentence1 + sentence2
            self.tokenizer.fit_on_texts(corpus)

        sentence1 = self.tokenizer.texts_to_sequences(sentence1)
        sentence1 = np.asarray(pad_sequences(sentence1, 
                                             padding=padding_type,
                                             truncating=truncating_type,
                                              maxlen=maxlen))
        
        sentence2 = self.tokenizer.texts_to_sequences(sentence2)
        sentence2 = np.asarray(pad_sequences(sentence2, 
                                             padding=padding_type,
                                             truncating=truncating_type,
                                             maxlen=maxlen))
        
        return sentence1, sentence2, self.tokenizer 




    # -------------------------------------------- Function to read embeddings -------------------------------------------- 
    
    def get_embedding_matrix(self, path, vocab, top=50000):
        """ Function to get the word embedding matrix.
        @param path (str): path to the word embeddings file.
        @param vocab (list): list of corpus vocab.
        @return embedding_matrix (np.array): embedding matrix.
        """
        if path.split(".")[-1] == "bin":
            embedding_model = KeyedVectors.load_word2vec_format(path, binary=True, limit=top)
        else:
            embedding_model = KeyedVectors.load_word2vec_format(path, binary=False, limit=top)
        
        embeddin_model_vocab =[word for word in embedding_model.vocab.keys()]
        final_vocab = list(set(vocab) | set(embeddin_model_vocab))      
        
        embedding_matrix = np.zeros((len(final_vocab), 300))
        cnt = 0
        for index in range(len(final_vocab)):
            if final_vocab[index] in embeddin_model_vocab:
                vec = embedding_model.wv[final_vocab[index]]
                if vec is not None:
                    embedding_matrix[index] = vec
            else:
                cnt = cnt + 1
                continue       
        print("zero embeddings: ", cnt)
        return embedding_matrix
        

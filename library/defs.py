#dependencies
import tensorflow as tf
import keras as ks
import pandas as pd
import sklearn as sk
import numpy as np
import random as rdm
import pathlib as plib
import os
import re

#definitions and classes
from keras import layers,models
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense,Dropout,GRU,LSTM
from keras.utils import pad_sequences
from nltk.corpus import stopwords
from keras.layers import Dense, Dropout, Embedding, GRU, LSTM, RNN, SpatialDropout1D

class NewsDetection(): 
    def print_csv(df): #simple print CSV
        print('CSV currently is, \n')
        print(df.head())
        print(df.tail())
        print(df.dtypes)
        print('\n')
    def fetch_csv(fetch_path,name): #fetching and loading the csv file from specified directory
        print('Fetching CSV file '+fetch_path+name+'.csv')
        df=read_csv(fetch_path+name+'.csv')
        print('Fetching complete.')
        return df
    def clean_string(string): #function to clean a string | to be used in clean_csv
        string=string.lower()
        string = re.sub('\n|\r|\t', '', string) #cleaning of whitespaces
        string = re.sub(r'[^\w\s]+', '', string) #cleaning of symbols
        return string
    def clean_csv(df): #cleaning the CSV file
        print('Cleaning CSV.')
        df=df.dropna() #drop all tuples with empty attributes
        length=[]
        [length.append(len(str(text))) for text in df["text"]] #calculate length of an article
        df['length'] = length #saves the length of the artivle in a new column
        df = df.drop(df['text'][df['length'] < 120].index, axis = 0) #drop all tuples with less than 120 letters
        #df['text']=df['title']+" "+df['text'] #append text to title
        vfunc=np.vectorize(clean_string) #vectorize th clean_string function for use in pandas df
        df['title']=vfunc(df['title']) #apply the vectorized function to further clean the data
        df['text']=vfunc(df['text']) #^^
        df.reset_index() #reset all indexes of the dataset
        print('Cleaning completed.')
        return df
    def preprocess_csv(df): #preprocessing the CSV file [depreceated]
        print('Preprocessing CSV.')
        corpus = []
        for i in range(0, len(df)):
            review = re.sub('[^a-zA-Z]', ' ', df['text'][i])
            review = review.lower()
            review = review.split()
            review = [word for word in review if not word in stopwords.words('english')]
            review = ' '.join(review)
            corpus.append(review)
        print('Preprocessed.')
        return corpus  
    def split_csv(df,test_prop): #splliting the dataset into test and train pieces based on proportion size
        print('Splitting the dataset based on proportion size of '+str(test_prop)+'.')
        x_train,x_test,y_train,y_test=train_test_split(df['text'],df['label'],test_size=test_prop,random_state=32,shuffle=True,stratify=None)
        y_train=pd.DataFrame(y_train)
        y_test=pd.DataFrame(y_test)
        print('Splitting successful.')
        return x_train,x_test,y_train,y_test
    def split_csv(df,df2,test_prop):
        print('Splitting the dataset based on proportion size of '+str(test_prop)+'.')
        x_train,x_test,y_train,y_test=train_test_split(df,df2,test_size=test_prop,random_state=32,shuffle=True,stratify=None)
        y_train=pd.DataFrame(y_train)
        y_test=pd.DataFrame(y_test)
        print('Splitting successful.')
        return x_train,x_test,y_train,y_test
    def create_tfidf(df): #creating a tf-idf doc for the NN to train on
        print('Creating TF-IDF Document.')
        tfidf_v=TfidfVectorizer(lowercase=True,max_df=0.7,min_df=0.2,use_idf=True,max_features=8000).fit(df)
        vocab=tfidf_v.vocabulary_
        #tfidf=tfidf_v.fit_transform(x.apply(lambda x: np.str_(x))) #lambda transforms normal str into numpy.str_
        tfidf_df=tfidf_v.transform(df)
        tfidf_df=pd.DataFrame(tfidf_df.toarray(),columns=sorted(vocab.keys()))
        #print(len(tfidf_df))
        print('Creation of TF-IDF Document finished.')
        return tfidf_df
    def create_cnn_model(df_len): #creating the internal structure of the CNN model
        print('Creating CNN model.')
        cnn_model=Sequential()
        cnn_model.add(Dense(128,input_shape=(194,),activation='relu'))
        cnn_model.add(Dropout(0.5))
        cnn_model.add(Dense(128,activation='relu'))
        cnn_model.add(Dropout(0.5))
        cnn_model.add(Dense(1,activation='sigmoid'))
        print('Model creation completed.')
        return cnn_model
    def compile_cnn_model(model): #compiling the created CNN model
        print('Compiling CNN model.')
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print('Compilation completed.')
        return model
    def train_cnn_model(model,tensor_train,y_train,tensor_test,y_test): #training the CNN model
        print('Training CNN model.')
        model.fit(x=tensor_train,y=y_train,batch_size=32,epochs=10,verbose=2 ,validation_data=(tensor_test, y_test))
        print('Training complete.')
        return model
    def create_lstm_model(max_features,len_in): #creating the internal structure of the LSTM model
        print('Creating LSTM model.')
        lstm_model = Sequential(name = 'lstm_nn_model')
        lstm_model.add(layer = Embedding(input_dim = max_features, output_dim = 120, name = '1st_layer'))
        lstm_model.add(layer = LSTM(units = 120, dropout = 0.2, recurrent_dropout = 0.2, name = '2nd_layer'))
        lstm_model.add(layer = Dropout(rate = 0.5, name = '3rd_layer'))
        lstm_model.add(layer = Dense(units = 120,  activation = 'relu', name = '4th_layer'))
        lstm_model.add(layer = Dropout(rate = 0.5, name = '5th_layer'))
        lstm_model.add(layer = Dense(units = (len_in),  activation = 'sigmoid', name = 'output_layer'))
        print('Model creation completed.')
        return lstm_model
    def compile_lstm_model(model): #compiling the created LSTM model
        print('Compiling LSTM model.')
        model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
        print('Compilation completed.')
        return model
    def train_lstm_model(model,X_train,Y_train): #training the LSTM model
        print('Training LSTM model.')
        model_fit = model.fit(X_train,Y_train,epochs=1)
        print('Training complete.')
        return model
    def save_model(model,model_path):
        print('Saving model.')
        model.save(model_path)
        print('Model saved.')
    
    def cache_df(df,cache_path,name): #caching df somewhere
        print('Caching DF to '+cache_path+name+'.csv')
        df.to_csv(cache_path+name+'.csv')
        print('DF cached.')
    def check_cache(cache_path,input_list): #check cache and return bool
        print('Checking cache for cached files...')
        files_list=[]
        for (dirpath, dirnames, filenames) in os.walk(cache_path):
            files_list.extend(filenames)
            break
        files_list.sort()
        #print(files_list)
        input_list.sort()
        #print(input_list)
        if len(files_list)==0:
            print('Cached files not found.')
            return False
        for i in range(0,len(input_list)):
            if input_list[i]+'.csv'==files_list[i]:
                print('Cached files found.')
                return True
        else:
            print('Cached files not found.')
            return False
    def pull_from_cache(cache_path,input_name): #pull pre-cached DFs from cache
        print('Pulling '+input_name+' from '+cache_path)
        df=read_csv(cache_path+input_name+'.csv')
        print('Pulled '+input_name+' from '+cache_path)
        return df
    def convert_df_to_tensor(df): #convert a pandas dataframe to tensorflow object for use in training the model
        print('Converting DF to Tensor object.')
        column_names = list(df.columns.values)
        converted_tensor = df[column_names]
        converted_tensor=tf.convert_to_tensor(converted_tensor)
        #print(converted_tensor)
        return converted_tensor
    def convert_df_to_numpy(df): #convert a pandas dataframe to a numpy array for use in training the model
        print('Converting DF to numpy object.')
        df=df.to_numpy()
        return df
    def tokenize_df(df,max_features): #tokneize the df using keras tokenizer
        print('Tokenizing DF.')
        tokenizer = Tokenizer(num_words = max_features, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', split = ' ')
        tokenizer.fit_on_texts(texts = df['text'])
        X = tokenizer.texts_to_sequences(texts = df['text'])
        X = pad_sequences(sequences = X, maxlen = max_features, padding = 'pre')
        #print(X.shape)
        Y = df['label'].values
        #print(Y.shape)
        print('Tokenized DF.')
        return X,Y
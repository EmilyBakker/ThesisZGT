import numpy as np
import pandas as pd

from tensorflow.keras import preprocessing
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


import re
import string

import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk import flatten
#from keras.preprocessing.text import Tokenizer
#from keras.preprocessing.sequence import pad_sequences




##############################

#GENERATOR TO SUPPLY DATA FOR MULTI MODAL MODEL

##############################

class MultiModalDataGenerator(Sequence):
    """
    Generator to supply data of all modalities
    """
    def __init__(self,
                label_data,                 # ID label index data
                demo_data: pd.DataFrame,    #demographic data
                vitals_data: pd.DataFrame,  #vitals data
                text_data: pd.DataFrame,    #text data
                batch_size=16,              #Set batch size for training sizes
                shuffle=True) :           #shuffle patient data over batches

        self.labels = pd.Series(label_data.target.values,index=label_data.Case_number).to_dict()

        self.list_IDs = label_data["Case_number"].tolist()
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.demo_data = demo_data
        self.vitals_data = vitals_data
        self.text_data = text_data

        self.on_epoch_end()                 #callback after each epoch
    
    def __len__(self):
        "Number of batches per epoch"
        return int(np.ceil(len(self.list_IDs) / self.batch_size))

    def on_epoch_end(self):
        "Conditionally shuffles indexes every epoch"
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def get_true_labels(self) -> list():
        list_IDs_temp = [self.list_IDs[k] for k in self.indexes]
        labels = [self.labels[ID] for ID in list_IDs_temp]

        return labels
    
    def get_ids(self) -> list:
        return [self.list_IDs[k] for k in self.indexes]

    def __getitem__(self, index):
        # Get IDs
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

         # Gather data from each modality
        demo_X = self.__get_demo(list_IDs_temp)       
        vitals_X = self.__get_vitals(list_IDs_temp)
        text_X = self.__get_text(list_IDs_temp)

        X = {
            'text' : text_X,
            'vitals': vitals_X,
            'demo': demo_X
        }

        y = self.__get_labels(list_IDs_temp)
        
        return X, y 

    def __get_demo(self, list_IDs_temp):
        #X_scaler = MinMaxScaler()
        X = list()

        df = self.demo_data.copy()
        #df['BMI'] = round(df['Gewicht']/((df['Lengte']/100)**2),1) #calculate BMI
        #df.drop(['Pt_nummer', 'Datum_operatie', 'Lengte','Gewicht','Pneumonie_datum','Naadlke_datum', 'geboortedatum','target_date'], axis=1, inplace=True) #drop irrelevant columns
        
        for ID in list_IDs_temp: #get data for each ID
            sample_data = df.loc[df['Case_number']==ID] #locate data in df
            sample_data = sample_data.drop(columns=['Case_number']) #drop Case_number
            #sample_data = X_scaler.fit_transform(sample_data[['Geslacht', 'operatiejaar', 'Leeftijd_OK', 'ASA']]) #normalize variables

            X.append(sample_data)
        
        X = np.array(X)
        x = X[:,:,:] #return 1d array, 2darray, and 3d array up to last column (to exclude the target variable)
        #y = X[:,:,-1] #return last column of most inner array
        
        return x
  
    def __get_vitals(self, list_IDs_temp): 
        X = list()

        df = self.vitals_data.copy()
        df = df.set_index('timestamp')
        df = df.sort_index()

        for ID in list_IDs_temp: 
            sample_data = df.loc[df['Case_number']==ID]
            sample_data = sample_data.drop(columns=['Case_number'])
            sample_data = sample_data.values.tolist()
    
            X.append(sample_data)
        
        # Pad inputs
        X = pad_sequences(X, padding='pre', dtype=float, value=-10.)
        x = X[:,:,:] #return 1d array, 2darray, and 3d array up to last column (to exclude the target variable)
        #y = X[:,:,-1] #return last column of most inner array 
        return x
 
    def __get_text(self, list_IDs_temp):
        text_out = list() #setp up list var
        ## limit the dataset to 50.000 words
        # The maximum number of words to be used
        max_n_words = 20000
        # Max number of words in each text
        max_text_length = 1000

        df = self.text_data.copy() #put df in var for cleaner code
        df = df.set_index('timestamp') #set index
        df = df.sort_index() #sort data in chronological order
        df.text = df.text.astype(str)

        tokenizer = Tokenizer(num_words=max_n_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True) #define tokenizer
        tokenizer.fit_on_texts(df['text'].values) #fit on text

        for ID in list_IDs_temp:
            sample_data = df.loc[df['Case_number']==ID] #get all observation for current case ID
            sample_data = sample_data.drop(columns=['Case_number']) #drop Case_number
            sample_data['text'] = tokenizer.texts_to_sequences(sample_data['text'].values) #tokenize text
            sample_data['text'] = pad_sequences(sample_data.text, maxlen=max_text_length, value=0).tolist() #pad texts to equal text length 
            sample_data =sample_data.values.tolist() # convert pandas.core.frame.DataFrame to list
            for i in range(len(sample_data)): #flatten arrays per observation
                sample_data[i]=flatten(sample_data[i])
            text_out.append(sample_data) #save data
        
        text_out = pad_sequences(text_out, padding='pre', dtype=float, value=-10.) #pad inputs
        x = text_out[:,:,:] #return 1d array, 2darray, and 3d array up to last column (data) (to exclude the target variable)
        #y = text_out[:,:,-1] #return last column of most inner array (labels)
        return x


    def __get_labels(self, list_IDs_temp):

        label_out = list()

        for ID in list_IDs_temp:
            temp_labels = self.labels.get(ID)

            if not isinstance(temp_labels, list):
                temp_labels=[temp_labels]
            
            label_out.append(temp_labels)

        return np.array(label_out)


class DemoVitalsDataGenerator(Sequence):
    """
    Generator to supply data of all modalities
    """
    def __init__(self,
                label_data,                 # ID label index data
                demo_data: pd.DataFrame,    #demographic data
                vitals_data: pd.DataFrame,  #vitals data
                batch_size=16,              #Set batch size for training sizes
                shuffle=True) :           #shuffle patient data over batches

        self.labels = pd.Series(label_data.target.values,index=label_data.Case_number).to_dict()

        self.list_IDs = label_data["Case_number"].tolist()
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.demo_data = demo_data
        self.vitals_data = vitals_data

        self.on_epoch_end()                 #callback after each epoch
    
    def __len__(self):
        "Number of batches per epoch"
        return int(np.ceil(len(self.list_IDs) / self.batch_size))

    def on_epoch_end(self):
        "Conditionally shuffles indexes every epoch"
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def get_true_labels(self) -> list():
        list_IDs_temp = [self.list_IDs[k] for k in self.indexes]
        labels = [self.labels[ID] for ID in list_IDs_temp]

        return labels
    
    def get_ids(self) -> list:
        return [self.list_IDs[k] for k in self.indexes]

    def __getitem__(self, index):
        # Get IDs
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

         # Gather data from each modality
        demo_X = self.__get_demo(list_IDs_temp)       
        vitals_X = self.__get_vitals(list_IDs_temp)

        X = {
            'vitals': vitals_X,
            'demo': demo_X
        }

        y = self.__get_labels(list_IDs_temp)
        
        return X, y 

    def __get_demo(self, list_IDs_temp):
        #X_scaler = MinMaxScaler()
        X = list()

        df = self.demo_data.copy()
        #df['BMI'] = round(df['Gewicht']/((df['Lengte']/100)**2),1) #calculate BMI
        #df.drop(['Pt_nummer', 'Datum_operatie', 'Lengte','Gewicht','Pneumonie_datum','Naadlke_datum', 'geboortedatum','target_date'], axis=1, inplace=True) #drop irrelevant columns
        
        for ID in list_IDs_temp: #get data for each ID
            sample_data = df.loc[df['Case_number']==ID] #locate data in df
            sample_data = sample_data.drop(columns=['Case_number']) #drop Case_number
            #sample_data = X_scaler.fit_transform(sample_data[['Geslacht', 'operatiejaar', 'Leeftijd_OK', 'ASA']]) #normalize variables

            X.append(sample_data)
        
        X = np.array(X)
        x = X[:,:,:] #return 1d array, 2darray, and 3d array up to last column (to exclude the target variable)
        #y = X[:,:,-1] #return last column of most inner array
        
        return x
  
    def __get_vitals(self, list_IDs_temp): 
        X = list()

        df = self.vitals_data.copy()
        df = df.set_index('timestamp')
        df = df.sort_index()

        for ID in list_IDs_temp: 
            sample_data = df.loc[df['Case_number']==ID]
            sample_data = sample_data.drop(columns=['Case_number'])
            sample_data = sample_data.values.tolist()
    
            X.append(sample_data)
        
        # Pad inputs
        X = pad_sequences(X, padding='pre', dtype=float, value=-10.)
        x = X[:,:,:] #return 1d array, 2darray, and 3d array up to last column (to exclude the target variable)
        #y = X[:,:,-1] #return last column of most inner array 
        return x
 
    
    def __get_labels(self, list_IDs_temp):

        label_out = list()

        for ID in list_IDs_temp:
            temp_labels = self.labels.get(ID)

            if not isinstance(temp_labels, list):
                temp_labels=[temp_labels]
            
            label_out.append(temp_labels)

        return np.array(label_out)
        


class DemoDataGenerator(Sequence):
    """
    Generator to supply data of demographic modality
    """
    def __init__(self, 
                label_data,
                demo_data: pd.DataFrame,
                batch_size=16,
                shuffle=True) -> None:

        self.labels = pd.Series(label_data.target.values,index=label_data.Case_number).to_dict()
        
        self.list_IDs = label_data["Case_number"].tolist()
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.demo_data = demo_data

        self.on_epoch_end() #callback after each epoch
    
    def __len__(self):
        "Number of batches per epoch"
        return int(np.ceil(len(self.list_IDs) / self.batch_size))

    def on_epoch_end(self):
        "Conditionally shuffles indexes every epoch"
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def get_true_labels(self) -> list():
        list_IDs_temp = [self.list_IDs[k] for k in self.indexes]
        labels = [self.labels[ID] for ID in list_IDs_temp]

        return labels
    
    def get_ids(self) -> list:
        return [self.list_IDs[k] for k in self.indexes]

    def __getitem__(self, index):
        # Get IDs
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

         # Gather data from each modality
        demo_X = self.__get_demo(list_IDs_temp) 

        

        X = {'demo': demo_X}

        y = self.__get_labels(list_IDs_temp)

        return X, y

    def __get_demo(self, list_IDs_temp):
        #X_scaler = MinMaxScaler()
        X = list()

        df = self.demo_data.copy()
        #df['BMI'] = round(df['Gewicht']/((df['Lengte']/100)**2),1) #calculate BMI
        #df.drop(['Pt_nummer', 'Datum_operatie', 'Lengte','Gewicht','Pneumonie_datum','Naadlke_datum', 'geboortedatum','target_date'], axis=1, inplace=True) #drop irrelevant columns
        
        for ID in list_IDs_temp: #get data for each ID
            sample_data = df.loc[df['Case_number']==ID] #locate data in df
            sample_data = sample_data.drop(columns=['Case_number']) #drop Case_number
            #sample_data = X_scaler.fit_transform(sample_data[['Geslacht', 'operatiejaar', 'Leeftijd_OK', 'ASA', 'BMI']]) #normalize variables

            X.append(sample_data)
        
        X = np.array(X)
        x = X[:,:,:] #return 1d array, 2darray, and 3d array up to last column (to exclude the target variable)
        #y = X[:,:,-1] #return last column of most inner array
        
        return x


    def __get_labels(self, list_IDs_temp):

        label_out = list()

        for ID in list_IDs_temp:
            temp_labels = self.labels.get(ID)

            if not isinstance(temp_labels, list):
                temp_labels=[temp_labels]
            
            label_out.append(temp_labels)

        return np.array(label_out)




class VitalsDataGenerator(Sequence):
    def __init__(self, 
                label_data,
                vitals_data: pd.DataFrame,
                batch_size=16,                    
                shuffle=True) -> None:

        self.vitals_data = vitals_data
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.labels = pd.Series(label_data.target.values,index=label_data.Case_number).to_dict()

        self.list_IDs = label_data["Case_number"].tolist()
        self.on_epoch_end()
        
    def __len__(self):
        "Number of batches per epoch"
        return int(np.ceil(len(self.list_IDs) / self.batch_size))

    def on_epoch_end(self):
        "Conditionally shuffles indexes every epoch"
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def get_true_labels(self) -> list():
        list_IDs_temp = [self.list_IDs[k] for k in self.indexes]
        labels = [self.labels[ID] for ID in list_IDs_temp]

        return labels

    def get_ids(self) -> list:
        return [self.list_IDs[k] for k in self.indexes]

    def __getitem__(self, index):
        # Get IDs
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

         # Gather data
        vitals_X = self.__get_vitals(list_IDs_temp)
        
        X = {'vitals': vitals_X}

        y = self.__get_labels(list_IDs_temp)
        
        return X, y
    
    def __get_vitals(self, list_IDs_temp): 
        X = list()

        df = self.vitals_data.copy()
        df = df.set_index('timestamp')
        df = df.sort_index()

        for ID in list_IDs_temp: 
            sample_data = df.loc[df['Case_number']==ID]
            sample_data = sample_data.drop(columns=['Case_number'])
            sample_data = sample_data.values.tolist()
    
            X.append(sample_data)
        
        # Pad inputs
        X = pad_sequences(X, padding='pre', dtype=float, value=-10.)
        x = X[:,:,:] #return 1d array, 2darray, and 3d array up to last column (to exclude the target variable)
        #y = X[:,:,-1] #return last column of most inner array 
        return x

    def __get_labels(self, list_IDs_temp):

        label_out = list()

        for ID in list_IDs_temp:
            temp_labels = self.labels.get(ID)

            if not isinstance(temp_labels, list):
                temp_labels=[temp_labels]
            
            label_out.append(temp_labels)

        return np.array(label_out)

class TextDataGenerator(Sequence):
    def __init__(self, 
                label_data,
                text_data,
                batch_size=16,                    
                shuffle=True) -> None:

        self.text_data = text_data
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.labels = pd.Series(label_data.target.values,index=label_data.Case_number).to_dict()

        self.list_IDs = label_data["Case_number"].tolist()
        self.on_epoch_end()
        
    def __len__(self):
        "Number of batches per epoch"
        return int(np.ceil(len(self.list_IDs) / self.batch_size))

    def on_epoch_end(self):
        "Conditionally shuffles indexes every epoch"
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def get_true_labels(self) -> list():
        list_IDs_temp = [self.list_IDs[k] for k in self.indexes]
        labels = [self.labels[ID] for ID in list_IDs_temp]

        return labels
    
    def get_ids(self) -> list:
        return [self.list_IDs[k] for k in self.indexes]

    def __getitem__(self, index):
        # Get IDs
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        #X_train[indices.astype(int)] # this is where I get the error
        #out_images = np.array(X_train)[indices.astype(int)]

         # Gather data
        text_X = self.__get_text(list_IDs_temp)
        
        X = {'text': text_X}

        y = self.__get_labels(list_IDs_temp)
        
        return X, y
    
    def __get_text(self, list_IDs_temp):
        X = list() #setp up list var
        ## limit the dataset to 50.000 words
        # The maximum number of words to be used
        max_n_words = 50000
        # Max number of words in each text
        max_text_length = 1000

        df = self.text_data.copy() #put df in var for cleaner code
        df = df.set_index('timestamp') #set index
        df = df.sort_index() #sort data in chronological order
        df.text=df.text.astype(str)

        tokenizer = Tokenizer(num_words=max_n_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True) #define tokenizer
        tokenizer.fit_on_texts(df['text'].values) #fit on text

        for ID in list_IDs_temp:
            sample_data = df.loc[df['Case_number']==ID] #get all observation for current case ID
            sample_data = sample_data.drop(columns=['Case_number']) #drop Case_number
            sample_data['text'] = tokenizer.texts_to_sequences(sample_data['text'].values) #tokenize text
            sample_data['text'] = pad_sequences(sample_data.text, maxlen=max_text_length, value=0).tolist() #pad texts to equal text length 
            sample_data =sample_data.values.tolist() # convert pandas.core.frame.DataFrame to list
            for i in range(len(sample_data)): #flatten arrays per observation
                sample_data[i]=flatten(sample_data[i])
            X.append(sample_data) #save data
        
        X = pad_sequences(X, padding='pre', dtype=float, value=-10.) #pad inputs
        #x = X[:,:,:] #return 1d array, 2darray, and 3d array up to last column (data) (to exclude the target variable)

        #y = text_out[:,:,-1] #return last column of most inner array (labels)
        return X

    def __get_labels(self, list_IDs_temp):

        label_out = list()

        for ID in list_IDs_temp:
            temp_labels = self.labels.get(ID)

            if not isinstance(temp_labels, list):
                temp_labels=[temp_labels]
            
            label_out.append(temp_labels)

        return np.array(label_out)





    

    
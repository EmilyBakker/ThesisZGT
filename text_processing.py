## Code to preprocess the text data for the model
#source: https://towardsdatascience.com/multi-class-text-classification-with-lstm-1590bee1bd17

##LIBS
import pandas as pd
import numpy as np
import re

import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize

nltk.download('wordnet', quiet=True)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# DATA
df = pd.read_csv("/mnt/data/embakker/long_text_sampled.csv", sep="|")

#STOPWORDS
stopwords = (stopwords.words("dutch"))


stopwords.remove('niet')
stopwords.remove('geen')

newStopWords = ['aangaande','aangezien','achter','arts','assistent','af','arts-assistent','achterna','afgelopen',
                'aldaar','althans','aldus','alhoewel','alle','allebei','alleen','alsnog','ander','behalve','beide', 
                'betreffende', 'binnen','boven','bovendien','io', 'i.o','i','o','mbv', 'mbt','ivm','nadien', 
                'bovenstaand','daarheen','daarin','daarna','daarnet','daarom','daarop','dikwijls','doorgaans',
                'echter','eerdat','eerder','eerst','elk','elke','enkele','enige','enz','etc','enzovoorts','etcetera', 
                'enigzinds','enkel','erdoor','even','eveneens','gauw','gedurende','hare','hierin','hierboven','hoewel', 
                'hijzelf','inmidddels','jezelf','jijzelf','jij','jou','jouw','juist','jullie','klaar','later','liever',
                'mag','mezelf','mijzelf','misschien','nabij','nadat','net','nogal','omhoog','omlaag','omtrent','ondertussen',
                'ongeveer','onszelf','opnieuw','opzij','overigens','pas','precies','rondom','sinds','sindsdien','slechts', '
                'sommige','spoedig','steeds','tamelijk','ten','tenzij','terwijl','tijdens','totdat','toe','tussen','uitgezonderd',
                'vaak','vandaan','vanuit','vanwege','verder','vervolgens','volgens','vooral','vooraf','vooralsnog',
                'voorbij','voordat','voorheen','vrij','vroeg','waar','waarom','wanneer','wegens','wel','wij','wijzelf','welke', 
                'zelfs','zodra','zowat','meneer','mevrouw','dokter','dr','hr','dhr','mvr', 'mv','mw','mevr','heer','vanmorgen', 
                'vanavond','vanmiddag', 'morgen', 'weer','ass','l','ltr', 'liter', 'obv' , 'cc', 'kouwenhoven','zichzelf','ws', 
                'waarschijnlijk','we','wij','waardoor','via','vd','vb','tov','tnt','tgv','ter','sv','svp','seldam','scu', 
                'ot','obs','mn','ml','mg','mgr','mililiter','miligram','leoniek', 'averdijk','irza','iom','ipv','det','artsass', 
                'cm','daarnaast','evt','akturk','dienst','crull','vh','vd']
stopwords.extend(newStopWords)
stopwords = set(stopwords)



#TEXT CLEANING
replace_by_space = re.compile('[/(){}\[\]\|@,;]') # to replace odd symbols by spacees
odd_symbols = re.compile('[^a-z]') #to recognize odd symbols (including digits and punctuation) '[^a-z #+_]'

def text_cleaner(txt):
    
    ## input: text data
    ## output: cleaned/modified text data
    
    txt = str(txt)
    txt = txt.lower() #transform to lower case
    txt = replace_by_space.sub(" ", txt) #substitute symbols with space
    txt = odd_symbols.sub(" ", txt) #substitute digits/punctuation with space
    #remove stopwords
    txt = ' '.join(word for word in txt.split() if word not in stopwords)
    
    return txt

df["text"] = df["text"].apply(text_cleaner)

#VECTORIZATION
## limit the dataset to 50.000 words
# The maximum number of words to be used
max_n_words = 50000
# Max number of words in each text
max_text_length = 1500
# set word embidding dimensionality
embedding_dim = 100

tokenizer = Tokenizer(num_words=max_n_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['text'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))





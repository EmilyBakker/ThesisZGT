import re
import pandas as pd

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk import word_tokenize

from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.model_selection import train_test_split

X_scaler = MinMaxScaler()

#####################################

#DEMOGRAPHIC DATA

#####################################
df_demo = pd.read_csv("/mnt/data/embakker/data_bin/overview_clean.csv", sep="|")
df_demo["BMI"] = round(df_demo["Gewicht"]/((df_demo["Lengte"]/100)**2),1)
df_demo.drop(['Pt_nummer', 'Datum_operatie', 'Lengte','Gewicht','Pneumonie_datum','Naadlke_datum', 'geboortedatum','target_date','target','complication_day'], axis=1, inplace=True)

#scale data
for column in df_demo[['Geslacht','operatiejaar','Leeftijd_OK','ASA','BMI']]:
    df_demo[[column]] = X_scaler.fit_transform(df_demo[[column]])

#write
df_demo.to_csv("/mnt/data/embakker/data/demo_all.csv", index=False, sep='|')


#####################################

#VITALS DATA

#####################################

df_vitals = pd.read_csv("/mnt/data/embakker/data_bin/long_structured_sampled.csv", sep="|")
df_vitals.drop("target", axis=1, inplace=True)

#encode categorical vars
for column in df_vitals[['measure']]:
    le = LabelEncoder().fit(df_vitals[[column]])
    df_vitals[[column]] = le.transform(df_vitals[[column]])

#scale data
for column in df_vitals[['day','measure', 'value']]:
    df_vitals[[column]] = X_scaler.fit_transform(df_vitals[[column]])

df_vitals.to_csv("/mnt/data/embakker/data/vitals_all.csv", index=False, sep="|")


#####################################

#TEXT DATA

#####################################

# Stopwords
stopwords = (stopwords.words("dutch"))
stopwords.remove('niet')
stopwords.remove('geen')
newStopWords = ['aangaande','aangezien','achter','arts','assistent','af','arts-assistent','achterna','afgelopen',
                'aldaar','althans','aldus','alhoewel','alle','allebei','alleen','alsnog','ander','behalve','beide',
                'betreffende','binnen','boven','bovendien','io', 'i.o','i','o','mbv', 'mbt','ivm','nadien', 
               'bovenstaand','daarheen','daarin','daarna','daarnet','daarom','daarop','dikwijls','doorgaans',
               'echter','eerdat','eerder','eerst','elk','elke','enkele','enige','enz','etc','enzovoorts','etcetera',
               'enigzinds','enkel','erdoor','even','eveneens','gauw','gedurende','hare','hierin','hierboven','hoewel',
               'hijzelf','inmidddels','jezelf','jijzelf','jij','jou','jouw','juist','jullie','klaar','later','liever',
               'mag','mezelf','mijzelf','misschien','nabij','nadat','net','nogal','omhoog','omlaag','omtrent','ondertussen',
               'ongeveer','onszelf','opnieuw','opzij','overigens','pas','precies','rondom','sinds','sindsdien','slechts',
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

def text_cleaner(txt):
    replace_by_space = re.compile('[/(){}\[\]\|@,;]') # to replace odd symbols by spacees
    odd_symbols = re.compile('[^a-z]') #to recognize odd symbols (including digits and punctuation) '[^a-z #+_]'

    ## input: text data
    ## output: cleaned/modified text data
    
    txt = str(txt)
    txt = txt.lower() #transform to lower case
    txt = replace_by_space.sub(" ", txt) #substitute symbols with space
    txt = odd_symbols.sub(" ", txt) #substitute digits/punctuation with space
    txt = ' '.join(word for word in txt.split() if word not in stopwords) #remove stopwords
    
    return txt




df_text = pd.read_csv("/mnt/data/embakker/data_bin/long_text_sampled.csv", sep="|")
df_text.drop("target", axis=1, inplace=True)
#df_text.Case_number =df_text.Case_number.astype(int)

df_text["text"] = df_text["text"].apply(text_cleaner)

#encode categorical vars
for column in df_text[['text_type']]:
    le = LabelEncoder().fit(df_text[[column]])
    df_text[[column]] = le.transform(df_text[[column]])

#scale data
for column in df_text[['day','text_type']]:
    df_text[[column]] = X_scaler.fit_transform(df_text[[column]])

df_text.to_csv("/mnt/data/embakker/data/text_all.csv", index=False, sep="|")

#####################################

#SPLITTING

#####################################

df_label = pd.read_csv("/mnt/data/embakker/data/labels.csv", sep=";")

X = df_label['Case_number'].values
Y = df_label['target'].values
X_train, X_test, Y_temp, Y_test = train_test_split(X, Y,stratify=Y, test_size=0.20)
X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test,stratify=Y_test, test_size=0.5)


df_demo_train = pd.DataFrame({'Case_number':X_train}).merge(df_demo)
df_demo_val = pd.DataFrame({'Case_number':X_val}).merge(df_demo)
df_demo_test = pd.DataFrame({'Case_number':X_test}).merge(df_demo)

df_vitals_train = pd.DataFrame({'Case_number':X_train}).merge(df_vitals)
df_vitals_val = pd.DataFrame({'Case_number':X_val}).merge(df_vitals)
df_vitals_test = pd.DataFrame({'Case_number':X_test}).merge(df_vitals)

df_label_train = pd.DataFrame({'Case_number':X_train}).merge(df_label)
df_label_val = pd.DataFrame({'Case_number':X_val}).merge(df_label)
df_label_test = pd.DataFrame({'Case_number':X_test}).merge(df_label)

#some dtypes in text are incompatible atm
#convert indexes to string
x_train = [str(x) for x in X_train]
x_val = [str(x) for x in X_val]
x_test = [str(x) for x in X_test]
df_text_train = pd.DataFrame({'Case_number':x_train}).merge(df_text)
df_text_val = pd.DataFrame({'Case_number':x_val}).merge(df_text)
df_text_test = pd.DataFrame({'Case_number':x_test}).merge(df_text)

#Write

df_demo_train.to_csv("/mnt/data/embakker/data/demo_train.csv", index=False, sep="|")
df_demo_val.to_csv("/mnt/data/embakker/data/demo_val.csv", index=False, sep="|")
df_demo_test.to_csv("/mnt/data/embakker/data/demo_test.csv", index=False, sep="|")

df_vitals_train.to_csv("/mnt/data/embakker/data/vitals_train.csv", index=False, sep="|")
df_vitals_val.to_csv("/mnt/data/embakker/data/vitals_val.csv", index=False, sep="|")
df_vitals_test.to_csv("/mnt/data/embakker/data/vitals_test.csv", index=False, sep="|")

df_text_train.to_csv("/mnt/data/embakker/data/text_train.csv", index=False, sep="|")
df_text_val.to_csv("/mnt/data/embakker/data/text_val.csv", index=False, sep="|")
df_text_test.to_csv("/mnt/data/embakker/data/text_test.csv", index=False, sep="|")

df_label_train.to_csv("/mnt/data/embakker/data/label_train.csv", index=False, sep="|")
df_label_val.to_csv("/mnt/data/embakker/data/label_val.csv", index=False, sep="|")
df_label_test.to_csv("/mnt/data/embakker/data/label_test.csv", index=False, sep="|")

print("Done preprocessing data")
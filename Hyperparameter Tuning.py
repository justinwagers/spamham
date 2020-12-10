import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from pprint import pprint
from kerastuner.tuners import RandomSearch
import regex as re 
from nltk.stem import WordNetLemmatizer
import string 
from nltk.corpus import stopwords

data = pd.read_csv('spam.csv')
lemma=WordNetLemmatizer()

data["Category"] = [1 if each == "spam" else 0 for each in data["Category"]]

def remove_symbols(text):
    text = re.sub('[^A-Za-z]', ' ', text)
    text = re.sub(r'(\\x(.){2})', '',text)
    return text
data['Message'] = data['Message'].apply(remove_symbols)

def lower_case(message):
    text = message.lower()  
    return text
data['Message'] = data['Message'].apply(lower_case)




def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    
    lemma = WordNetLemmatizer()
    nopunc = [ lemma.lemmatize(word) for word in nopunc]
    
data['Message'] = data['Message'].apply(text_process)

def rejoin_words(row):
    row
    joined_words = ( " ".join(row))
    return joined_words
data['Message'] = data['Message'].apply(rejoin_words)


labels = data.iloc[:, 0]


  


from keras.layers import SimpleRNN, Embedding, Dense, LSTM
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
max_features = 10000
maxlen = 500


texts = []
for i, label in enumerate(data['Category']):
    texts.append(data['Message'][i])

texts = np.asarray(texts)
labels = np.asarray(labels)
from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences_tfidf = tokenizer.texts_to_matrix(data['Message'], mode='tfidf')

X = pad_sequences(sequences_tfidf, maxlen=maxlen)
sequences = tokenizer.texts_to_sequences(texts)

X = pad_sequences(sequences, maxlen=maxlen)
tfidf = tokenizer.texts_to_matrix(data['Message'], mode='tfidf')
X_tfidf = pad_sequences(tfidf, maxlen=maxlen)


X_train, X_test, y_train, y_test = train_test_split(X_tfidf,labels, test_size = 0.2, random_state = 42)


def tune_optimizer_model(hp):
    model = Sequential()
    model.add(Embedding(max_features, 32))
    #model.add(SimpleRNN(32))
    model.add(keras.layers.SimpleRNN(units=hp.Int('units',
                                        min_value=8,
                                        max_value=128,
                                        step=16),
                                 activation="relu", 
                                 input_shape=[X_train.shape[1]]))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = hp.Choice('optimizer', ['adam', 'sgd', 'rmsprop'])

    model.compile(
        optimizer=optimizer,
        loss = 'binary_crossentropy', 
        metrics = ['accuracy'])
    return model

def tune_neurons_model(hp):
    model = Sequential()
    model.add(Embedding(max_features, 32))
    #model.add(SimpleRNN(32))
    model.add(keras.layers.SimpleRNN(units=hp.Int('units',
                                        min_value=16,
                                        max_value=128,
                                        step=16),
                                 activation="relu", 
                                 input_shape=[X_train.shape[1]]))
    model.add(Dense(1, activation='sigmoid'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.compile(
        optimizer="sgd",
        loss = 'binary_crossentropy', 
        metrics = ['accuracy'])
    return model

def create_random_tuner(model_builder, project_name):
  tuner = RandomSearch(
    model_builder,
    objective='val_accuracy',
    max_trials=MAX_TRIALS,
    executions_per_trial=EXECUTIONS_PER_TRIAL,
    directory=os.path.normpath('C:/'), 
    project_name=project_name,
    seed=RANDOM_SEED
  )




  tuner.search_space_summary()
  return tuner

def random_search_params(model_builder, project_name):
  tuner = create_random_tuner(model_builder, project_name)
  tuner.search(x=X_train,
             y=y_train,
             epochs=TRAIN_EPOCHS,
             validation_data=(X_test, y_test))
  
  tuner.results_summary()

  pprint(tuner.oracle.get_best_trials(num_trials=1)[0].hyperparameters.values)

random_search_params(tune_neurons_model, project_name="tune_neurons3")



MAX_TRIALS = 10
EXECUTIONS_PER_TRIAL = 5

RANDOM_SEED = 21
tuner = RandomSearch(
    tune_optimizer_model,
    objective='val_accuracy',
    max_trials=MAX_TRIALS,
    executions_per_trial=EXECUTIONS_PER_TRIAL,
    directory= os.path.normpath('C:/'), 
    project_name='tune_optimizer2',
    seed=RANDOM_SEED
)

tuner.search_space_summary()

TRAIN_EPOCHS = 20

tuner.search(x=X_train,
             y=y_train,
             epochs=TRAIN_EPOCHS,
             validation_data=(X_test, y_test))

def tune_act_model(hp):
    model = keras.Sequential()

    activation = hp.Choice('activation', 
                            ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear'])
    optimizer = hp.Choice('optimizer', ['adam', 'sgd', 'rmsprop'])

    model.add(Embedding(max_features, 32))
    model.add(keras.layers.SimpleRNN(units=hp.Int('units',
                                        min_value=8,
                                        max_value=128,
                                        step=16),
                                 activation=activation, 
                                 input_shape=[X_train.shape[1]]))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(
        optimizer=optimizer,
        loss = 'binary_crossentropy', 
        metrics = ['accuracy'])
    return model



random_search_params(tune_act_model, "tune_activation_RNNTFIDF1")


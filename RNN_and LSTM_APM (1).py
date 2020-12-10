# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 17:36:09 2020

@author: machu
"""
from sklearn.metrics import roc_curve,roc_auc_score

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import TfidfModel 
import regex as re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize as wt

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import string
import numpy as np
from tensorflow.keras.layers import Dropout

#Bag of Words Simple RNN 


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
sequences = tokenizer.texts_to_sequences(texts)


X = pad_sequences(sequences, maxlen=maxlen)

X_train, X_test, y_train, y_test = train_test_split(X,labels, test_size = 0.2, random_state = 42)

model_rnn = Sequential()
model_rnn.add(Embedding(max_features, 64))
model_rnn.add(SimpleRNN(units= 56,
                                 activation='tanh', 
                                 input_shape=[X_train.shape[1]]))
model_rnn.add(Dense(1, activation='sigmoid'))

model_rnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
history_rnn = model_rnn.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

model_rnn.summary()
import matplotlib.pyplot as plt
acc = history_rnn.history['acc']
val_acc = history_rnn.history['val_acc']
loss = history_rnn.history['loss']
val_loss = history_rnn.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, '-', color='orange', label='training acc')
plt.plot(epochs, val_acc, '-', color='blue', label='validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()

plt.plot(epochs, loss, '-', color='orange', label='training acc')
plt.plot(epochs, val_loss,  '-', color='blue', label='validation acc')
plt.title('Training and validation loss')
plt.legend()
plt.show()

pred = model_rnn.predict_classes(X_test)
pred_train = model_rnn.predict_classes(X_train)
acc = model_rnn.evaluate(X_test, y_test)
acc_train = model_rnn.evaluate(X_train, y_train)
proba_rnn = model_rnn.predict_proba(X_test)
from sklearn.metrics import confusion_matrix
print("Test loss is {0:.2f} accuracy is {1:.2f}  ".format(acc[0],acc[1]))
print(confusion_matrix(pred, y_test))
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score,roc_curve

cr = classification_report(y_test, pred)
print(cr)
cm = confusion_matrix(y_test,pred)
print(cm)
print(acc)
print(acc_train)
res = pd.get_dummies(y_test)

res[0]



fpr , tpr , thresholds = roc_curve( res[1], proba_rnn)

def plot_roc_curve(fpr,tpr): 
    plt.plot(fpr,tpr) 
    plt.axis([0,1,0,1]) 
    plt.xlabel('False Positive Rate') 
    plt.ylabel('True Positive Rate') 
    plt.title('RNN ROC curve')
    plt.show()



plot_roc_curve (fpr,tpr) 

class_names = ['ham', 'spam']
import itertools
import matplotlib.pyplot as plt
def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


plot_confusion_matrix(cm=cm, classes=class_names, title='Simple RNN BoW Confusion Matrix')

## Bag of Words LSTM 

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
sequences = tokenizer.texts_to_sequences(texts)

X = pad_sequences(sequences, maxlen=maxlen)

X_train, X_test, y_train, y_test = train_test_split(X,labels, test_size = 0.2, random_state = 42)

model_LSTM = Sequential()
model_LSTM.add(Embedding(max_features, 64))
model_LSTM.add(LSTM(units= 24,
                                 activation='tanh', 
                                 input_shape=[X_train.shape[1]]))
model_LSTM.add(Dense(1, activation='sigmoid'))

model_LSTM.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
history_rnn = model_LSTM.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

import matplotlib.pyplot as plt
acc = history_rnn.history['acc']
val_acc = history_rnn.history['val_acc']
loss = history_rnn.history['loss']
val_loss = history_rnn.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, '-', color='orange', label='training acc')
plt.plot(epochs, val_acc, '-', color='blue', label='validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()

plt.plot(epochs, loss, '-', color='orange', label='training acc')
plt.plot(epochs, val_loss,  '-', color='blue', label='validation acc')
plt.title('Training and validation loss')
plt.legend()
plt.show()


model_LSTM.summary()
pred_LSTM = model_LSTM.predict_classes(X_test)
pred_LSTM_train = model_LSTM.predict_classes(X_train)
acc_tf = model_LSTM.evaluate(X_test, y_test)
acc_tf_train = model_LSTM.evaluate(X_train, y_train)
proba_lstm = model_LSTM.predict_proba(X_test)
from sklearn.metrics import confusion_matrix
print("Test loss is {0:.2f} accuracy is {1:.2f}  ".format(acc_tf[0],acc_tf[1]))
print(confusion_matrix(pred_LSTM, y_test))
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

cr_tf = classification_report(y_test, pred_LSTM)
print(cr_tf)
print(acc_tf)
print(acc_tf_train)
cm = confusion_matrix(pred_LSTM, y_test)
res = pd.get_dummies(y_test)

res[0]


y_val_cat_prob = proba_lstm
fpr , tpr , thresholds = roc_curve ( res[1], proba_lstm)

def plot_roc_curve(fpr,tpr): 
    plt.plot(fpr,tpr) 
    plt.axis([0,1,0,1]) 
    plt.xlabel('False Positive Rate') 
    plt.ylabel('True Positive Rate') 
    plt.title('LSTM ROC curve')
    plt.show()



plot_roc_curve (fpr,tpr) 
plot_confusion_matrix(cm=cm, classes=class_names, title='LSTM BoW Confusion Matrix')



#TFIDF RNN 

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
sequences = tokenizer.texts_to_matrix(data['Message'], mode='tfidf')

X = pad_sequences(sequences, maxlen=maxlen)

X_train, X_test, y_train, y_test = train_test_split(X,labels, test_size = 0.2, random_state = 42)


model_tfidf_RNN = Sequential()
model_tfidf_RNN.add(Embedding(max_features, 32))
model_tfidf_RNN.add(SimpleRNN(units= 120,
                                 activation='softsign', 
                                 input_shape=[X_train.shape[1]]))
model_tfidf_RNN.add(Dense(1, activation='sigmoid'))

model_tfidf_RNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
history_rnn = model_tfidf_RNN.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
print(model_tfidf_RNN.summary())
import matplotlib.pyplot as plt
acc = history_rnn.history['acc']
val_acc = history_rnn.history['val_acc']
loss = history_rnn.history['loss']
val_loss = history_rnn.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, '-', color='orange', label='training acc')
plt.plot(epochs, val_acc, '-', color='blue', label='validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()

plt.plot(epochs, loss, '-', color='orange', label='training acc')
plt.plot(epochs, val_loss,  '-', color='blue', label='validation acc')
plt.title('Training and validation loss')
plt.legend()
plt.show()



pred_tfidf = model_tfidf_RNN.predict_classes(X_test)
pred_tfidf = model_tfidf_RNN.predict_classes(X_train)
acc_tfidf = model_tfidf_RNN.evaluate(X_test, y_test)
acc_tfidf_train = model_tfidf_RNN.evaluate(X_train, y_train)


proba_tfidf = model_tfidf_RNN.predict_proba(X_test)
from sklearn.metrics import confusion_matrix
print("Test loss is {0:.2f} accuracy is {1:.2f}  ".format(acc_tfidf[0],acc_tfidf[1]))
print(confusion_matrix(pred_tfidf, y_test))
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

cr_tfidf = classification_report(y_test, pred_tfidf)
print(cr_tfidf)
print(acc_tfidf)
print(acc_tfidf_train)
cm_tfidf = confusion_matrix(pred_tfidf,y_test)
import matplotlib.pyplot as plt
y_val_cat_prob=pred_tfidf

from sklearn.metrics import roc_curve,roc_auc_score





res = pd.get_dummies(y_test)

res[0]



fpr , tpr , thresholds = roc_curve ( res[1], proba_tfidf)

def plot_roc_curve(fpr,tpr): 
    plt.plot(fpr,tpr) 
    plt.axis([0,1,0,1]) 
    plt.xlabel('False Positive Rate') 
    plt.ylabel('True Positive Rate') 
    plt.title('RNN TF-IDF ROC curve')
    plt.show()



plot_roc_curve (fpr,tpr) 

plot_confusion_matrix(cm=cm_tfidf, classes=class_names, title='Simple RNN TFIDF Confusion Matrix')

# TFIDF LSTM 

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
sequences = tokenizer.texts_to_matrix(data['Message'], mode='tfidf')

X = pad_sequences(sequences, maxlen=maxlen)

X_train, X_test, y_train, y_test = train_test_split(X,labels, test_size = 0.2, random_state = 42)


model_lstm_tfidf = Sequential()
model_lstm_tfidf.add(Embedding(max_features, 64))
model_lstm_tfidf.add(LSTM(units= 120,
                                 activation='softsign', 
                                 input_shape=[X_train.shape[1]]))
model_lstm_tfidf.add(Dense(1, activation='sigmoid'))

model_lstm_tfidf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
history_rnn = model_lstm_tfidf.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
model_lstm_tfidf.summary()
import matplotlib.pyplot as plt
acc = history_rnn.history['acc']
val_acc = history_rnn.history['val_acc']
loss = history_rnn.history['loss']
val_loss = history_rnn.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, '-', color='orange', label='training acc')
plt.plot(epochs, val_acc, '-', color='blue', label='validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()

plt.plot(epochs, loss, '-', color='orange', label='training acc')
plt.plot(epochs, val_loss,  '-', color='blue', label='validation acc')
plt.title('Training and validation loss')
plt.legend()
plt.show()



pred_LSTM_tfidf = model_lstm_tfidf.predict_classes(X_test)
pred_LSTM_tfidf_train = model_lstm_tfidf.predict_classes(X_train)

acc_tfidf_lstm  = model_lstm_tfidf.evaluate(X_test, y_test)
acc_tfidf_lstm_train  = model_lstm_tfidf.evaluate(X_train, y_train)

proba_ltsm_tfidf = model_lstm_tfidf.predict_proba(X_test)
from sklearn.metrics import confusion_matrix
print("Test loss is {0:.2f} accuracy is {1:.2f}  ".format(acc_tfidf_lstm[0],acc_tfidf_lstm[1]))
print(confusion_matrix(pred_LSTM_tfidf, y_test))
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import plot_roc_curve

cr_tfidf_lstm = classification_report(y_test, pred_LSTM_tfidf)
print(cr_tfidf_lstm)
print(acc_tfidf_lstm)
print(acc_tfidf_lstm_train)

cm_tfidf = confusion_matrix(pred_LSTM_tfidf, y_test)

res = pd.get_dummies(y_test)

res[0]



fpr , tpr , thresholds = roc_curve ( res[1], proba_ltsm_tfidf)

def plot_roc_curve(fpr,tpr): 
    plt.plot(fpr,tpr) 
    plt.axis([0,1,0,1]) 
    plt.xlabel('False Positive Rate') 
    plt.ylabel('True Positive Rate') 
    plt.title('LSTM TF-IDF ROC curve')
    plt.show()



plot_roc_curve (fpr,tpr) 

def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



plot_confusion_matrix(cm=cm, classes=class_names, title='Multinomial Bayes TFIDF Confusion Matrix')

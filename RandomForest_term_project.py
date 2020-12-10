# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 09:50:05 2020

@author: machu
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split




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


from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
bow_transformer = CountVectorizer(analyzer=text_process).fit(data['Message'])
messages_bow = bow_transformer.transform(data['Message'])

tfidf_transformer = TfidfTransformer().fit(messages_bow)
messages_tfidf = tfidf_transformer.transform(messages_bow)
  

X_train, X_test, y_train, y_test = train_test_split(messages_tfidf,labels, test_size = 0.2, random_state = 42)



rf = RandomForestClassifier(n_estimators=100,max_depth=None,n_jobs=-1)
rf_model = rf.fit(X_train,y_train)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
rf_pred =  rf_model.predict(X_test)

print(classification_report(rf_pred,y_test))
print(confusion_matrix(rf_pred,y_test))
accuracy = accuracy_score(y_test, rf_pred)
print(accuracy)


from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 

n_estimators = [int(x) for x in range(100,2000,200)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid

def evaluate(model, test_features, test_labels):
    y_pred = model.predict(test_features)
    accuracy = accuracy_score(y_test, y_pred)
    print (accuracy)
    print(confusion_matrix(y_test,y_pred))
    




param_grid = {
    'bootstrap': [True],
    'max_depth': [10,15,None],
    'max_features': [2, 3,'auto'],
    'min_samples_leaf': [1, 4, 5,6],
    'min_samples_split': [2,4,5,6],
    'n_estimators': [100, 200, 300, 400, 500]
}
# Create a based model
rf = RandomForestClassifier(random_state = 42)
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)

grid_search.fit(X_train,y_train)

gridsearch_pred =  grid_search.predict(X_test)
gridsearch_pred_train = grid_search.predict(X_train)

grid_search.best_params_



best_grid = grid_search.best_estimator_
evaluate(best_grid,X_test,y_test)
accuracy = accuracy_score(y_test, gridsearch_pred)
accuracy_train = accuracy_score(y_train, gridsearch_pred_train)

print(accuracy)
print(accuracy_train)

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

cm_rf = confusion_matrix(y_test,gridsearch_pred)
plot_confusion_matrix(cm=cm_rf, classes=class_names, title='Random Forest Confusion Matrix')



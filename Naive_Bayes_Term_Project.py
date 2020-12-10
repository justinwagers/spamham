# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 15:38:51 2020

@author: machu
"""
from nltk.corpus import stopwords
import string
import pandas as pd
from nltk.stem import WordNetLemmatizer
import nltk 
import regex as re
from autocorrect import spell
import re
import nltk
from sklearn.naive_bayes import MultinomialNB

nltk.download('punkt')
from nltk.tokenize import word_tokenize as wt


data = pd.read_csv('spam.csv')
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

    
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
labels = data.iloc[:, 0]

X1_train, X1_test, y1_train, y1_test = train_test_split(data['Message'] ,labels, test_size = 0.2, random_state = 42)

from sklearn.model_selection import train_test_split

classifier = MultinomialNB()



from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB())])

tuned_parameters = {
    'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],
    'tfidf__use_idf': (True, False),
    'tfidf__norm': ('l1', 'l2'),
    'clf__alpha': [1, 1e-1, 1e-2]
}



clf = GridSearchCV(text_clf, tuned_parameters, cv=10, scoring="accuracy")
clf.fit(X1_train, y1_train)
pred = clf.predict(X1_test)
pred_train = clf.predict(X1_train)
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
print(classification_report(y1_test, clf.predict(X1_test), digits=4))
print(confusion_matrix(y1_test, clf.predict(X1_test)))
accuracy = accuracy_score(pred, y1_test)
accuracy_train = accuracy_score(pred_train, y1_train)
print(accuracy)
print(accuracy_train)
print("Best parameters set:")
print(clf.best_estimator_.steps)
import matplotlib.pyplot as plt
import sklearn.metrics as metrics1
fpr, tpr, threshold = metrics1.roc_curve(y1_test, pred)
roc_auc = metrics1.auc(fpr, tpr)

# method I: plt
import matplotlib.pyplot as plt
plt.title('Multinomial Naive Bayes')
plt.plot(fpr, tpr)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()






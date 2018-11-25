# ML imports

from sklearn.naive_bayes import MultinomialNB

# from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import TfidfVectorizer

# from sklearn.ensemble import RandomForestClassifier
import pickle

from util import plot_roc


class NLPModel(object):

    def __init__(self):
        '''
        keeping simple NLP attributes

        clf: sklearn classifier model
        vectorizer: TFIDF vectorizer
        '''

        self.clf = MultinomialNB()
        self.vectorizer = TfidfVectorizer()

    def vectorizer_fit(self,X):
        # fitting the Tfidfvectorizer to the text
        self.vectorizer.fit(X)

    def vectorizer_transform(self,X):
        # Transform the text data to a sparse TFIDF matrix
        X_transformed = self.vectorizer.transform(X)
        return X_transformed

    def train(self,X,y):
        self.clf.fit(X,y)

    def predict_proba(self,X):
        y_proba  = self.clf.predict_proba(X)
        return y_proba[:,1]

    def predict(self,X):
        y_pred = self.clf.predict(X)
        return y_pred

    def pickle_vectorizer(self,path='TFIDFVectorizer.pkl'):

        with open(path,'wb') as f:
            pickle.dump(self.vectorizer,f)
            print('Pickled vectorizer at {}'.format(path))

    def pickle_clf(self,path='SentimentClassifier.pkl'):

        with open(path,'wb') as f:
            pickle.dump(self.clf, f)
            print('Pickled classifier at {}'.format(path))


    def plot_roc(self,X,y,size_x, size_y):
        plot_roc(self.clf,X,y,size_x,size_y)


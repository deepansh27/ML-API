from model import NLPModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def build_model():
    model = NLPModel()
    with open('./data/train.tsv') as f:
        data= pd.read_csv(f,sep='\t')

    print(data.columns)
    pos_neg = data[(data['Sentiment'] == 0) | (data['Sentiment'] == 4)]

    pos_neg['Binary'] = np.where(pos_neg['Sentiment'] == 0, 0, 1)

    model.vectorizer_fit(pos_neg.loc[:, 'Phrase'])
    print('Vectorizer fit complete')

    X = model.vectorizer_transform(pos_neg.loc[:, 'Phrase'])
    print('Vectorizer transform complete')
    y = pos_neg.loc[:, 'Binary']

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model.train(X_train, y_train)
    print('Model training complete')

    model.pickle_clf()
    model.pickle_vectorizer()

    # model.plot_roc(X_test, y_test)


if __name__ == "__main__":
    build_model()
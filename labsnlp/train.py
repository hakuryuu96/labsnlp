import pandas as pd
from CONFIG import *
from preprocessing import preprocess_tags, preprocess_text
from vectorize import get_tfidf_vectorizer
from model import train_gaussian_nb_classifier

from sklearn.metrics import f1_score
from joblib import dump, load

def train():
    train_data = pd.read_csv(TRAIN_DATA_PATH, index_col=0)
    test_data = pd.read_csv(TEST_DATA_PATH, index_col=0)

    preprocessed_tags_train_data = preprocess_tags(train_data)
    preprocessed_tags_test_data = preprocess_tags(test_data)

    preprocessed_text_train_data = preprocess_text(preprocessed_tags_train_data)
    preprocessed_text_test_data = preprocess_text(preprocessed_tags_test_data)

    vectorizer = get_tfidf_vectorizer(preprocessed_text_train_data['plot_synopsis'].values.tolist(), max_features=10000)

    # sparse
    X_train = vectorizer.transform(preprocessed_text_train_data['plot_synopsis'].values.tolist())
    X_test = vectorizer.transform(preprocessed_text_test_data['plot_synopsis'].values.tolist())

    # only for gothic
    clf = train_gaussian_nb_classifier(X_train.toarray(), train_data['is_gothic'].values.tolist())

    predictions = clf.predict(X_test.toarray())

    print(f'Micro F1: {f1_score(test_data["is_gothic"].values.tolist(), predictions, average="micro")}')

    # saving vectorizer and model
    dump(clf, 'clf.joblib') 
    dump(vectorizer, 'vectorizer.joblib') 


def prepare_submission():
    clf = load('clf.joblib')
    vectorizer = load('vectorizer.joblib')

    data = pd.read_csv(SUBMISSION_DATA_TEST_PATH, index_col=0)
    preprocessed_data = preprocess_text(data)

    X = vectorizer.transform(preprocessed_data['plot_synopsis'].values.tolist())
    predictions = clf.predict(X.toarray()).tolist()

    data['tags'] = ['gothic' if x == 1 else 'another' for x in predictions]
    data['tags'].to_csv('submission.csv', sep=';')


if __name__ == "__main__":
    train()
    prepare_submission()
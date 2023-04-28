import pandas as pd

import nltk
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def preprocess_tags(data):
    tags = data['tags']

    unique_tags = list(set(tag for row in tags for tag in row.split(', ')))
    for tag in unique_tags:
        data[f'is_{tag}'] = data['tags'].apply(lambda x: 1 if tag in x.split(', ') else 0)
    
    return data


def preprocess_text(data):
    # removing signs, leaving only words 
    print('Preprocessing text')
    print('Removing signs, leaving only words ')
    data['plot_synopsis'] = data['plot_synopsis'].str.lower(). \
                                                  str.replace(r'[^\w\s]+', '', regex=True). \
                                                  str.replace(r'\s+', ' ', regex=True).\
                                                  str.strip()
    
    # remove stopwords
    print('removing stopwords')
    stopwords_list = stopwords.words('english')
    data['plot_synopsis'] = data['plot_synopsis'].apply(lambda x: " ".join([word for word in x.split(' ') if word not in stopwords_list]))

    # lemmatize 
    print('lemmatizing')
    lemmatizer = WordNetLemmatizer()
    data['plot_synopsis'] = data['plot_synopsis'].apply(lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split(' ')]))

    return data
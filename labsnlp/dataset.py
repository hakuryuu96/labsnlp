import torch
from torch.utils.data import Dataset

import os
import pandas as pd

from preprocessing import preprocess_tags, preprocess_text
import numpy as np


def pad_seuqence(vector, length):
    if vector.shape[0] < length:
        while vector.shape[0] < length:
            vector = torch.cat([vector, vector], dim=0)
    return vector[:length]


class SentenceDataset(Dataset):
    def __init__(
        self,
        annotation_file: str,
        embeddings_file: str,
        limit_numvectors: int = 50
    ):
        super().__init__()

        assert os.path.exists(annotation_file)
        assert os.path.exists(embeddings_file)

        self.annotation_file = annotation_file
        self.embeddings_file = embeddings_file

        self.data = pd.read_csv(annotation_file, index_col=0)
        self.__prepare_data()

        self.word2idx = {}
        self.word_vectors = []
        self.__load_embeddings()

        self.limit_numvectors = limit_numvectors
    
    def __prepare_data(self):
        preprocessed_tags = preprocess_tags(self.data)
        preprocessed_text = preprocess_text(preprocessed_tags)

        self.data = preprocessed_text

    def __load_embeddings(self):
        idx = 0
        with open(self.embeddings_file, 'rb') as f:
            for l in f:
                line = l.decode().split()
                word = line[0]
                self.word2idx[word] = idx
                idx += 1
                vect = np.array(line[1:]).astype(np.float64)
                self.word_vectors.append(vect)
            
            f.close()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        curr_data = self.data.iloc[idx]
        
        plot_synopsis = curr_data.plot_synopsis
        classes = np.array(
            [curr_data[x] for x in self.data.columns if 'is_' in x]
        )

        vectors = []
        for word in plot_synopsis.split(' '):
            if word in self.word2idx.keys():
                vectors.append(
                    self.word_vectors[self.word2idx[word]]
                )
        
        vectors = np.stack(vectors)

        vectors = torch.from_numpy(vectors).to(dtype=torch.float)
        classes = torch.from_numpy(classes).to(dtype=torch.float)

        vectors = pad_seuqence(vectors, self.limit_numvectors)

        return {
            'word_vectors': vectors,
            'classes': classes
        }

if __name__ == "__main__":
    s = SentenceDataset(
        annotation_file='../data/public_train_train.csv',
        embeddings_file='../data/glove.6B/glove.6B.50d.txt'
    )

    print(s[0])
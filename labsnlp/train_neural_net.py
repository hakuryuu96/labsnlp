import pandas as pd
from CONFIG import *

from nn_model import MyLSTMClassifier
from dataset import SentenceDataset

# from sklearn.metrics import f1_score

from torchmetrics.classification import MultilabelF1Score

import torch
import torch.nn as nn

import tqdm


N_EPOCHS = 20
NUM_CLASSES = 71
EMBEDDING_DIM = 50
HIDDEN_DIM = NUM_CLASSES // 2
BS = 1

LR = 1e-1
WD = 1e-4
LIMIT_NUMVECTORS = 100

def train():
    train_dataset = SentenceDataset(
        TRAIN_DATA_PATH,
        PATH_TO_VECTOR,
        limit_numvectors=LIMIT_NUMVECTORS
    )

    test_dataset = SentenceDataset(
        TEST_DATA_PATH,
        PATH_TO_VECTOR,
        limit_numvectors=LIMIT_NUMVECTORS
    )

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BS, shuffle=False)

    model = MyLSTMClassifier(
        num_classes = NUM_CLASSES, 
        embedding_dim = EMBEDDING_DIM, 
        hidden_dim = HIDDEN_DIM,
    )

    loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

    max_metric = 0

    for epoch in range(N_EPOCHS): 
        print(f'Epoch {epoch}...') 
        print('--> Training...')
        for element in tqdm.tqdm(train_dataloader):
            model.zero_grad()

            sentence = element['word_vectors']
            classes = element['classes']
            pred_classes = model(sentence)

            loss = loss_function(pred_classes, classes)
            loss.backward()
            optimizer.step()
        
        print('--> Validation...')
       
        metric = MultilabelF1Score(
            NUM_CLASSES,
            average='micro'
        )

        predictions = []
        targets = []

        with torch.no_grad():
            for element in tqdm.tqdm(test_dataset):
                sentence = element['word_vectors']
                classes = element['classes']

                pred_classes = model(sentence)

                predictions.append(pred_classes)
                targets.append(classes)
        
        predictions = torch.stack(predictions)
        targets = torch.stack(targets)


        curr_metric = metric(predictions, targets)
        print(f'--> Micro F1: {curr_metric}')

        if curr_metric > max_metric:
            print('--> Current best... Saving as checkpoint.')
            max_metric = curr_metric

            torch.save(model.state_dict(), f'best_checkpoint.pt')


def prepare_submission():
    state_dict = torch.load('best_checkpoint.pt')
    model = MyLSTMClassifier(
        num_classes = NUM_CLASSES, 
        embedding_dim = EMBEDDING_DIM, 
        hidden_dim = HIDDEN_DIM,
    )
    model.load_state_dict(state_dict)

    # ...

    # data = pd.read_csv(SUBMISSION_DATA_TEST_PATH, index_col=0)
    # preprocessed_data = preprocess_text(data)

    # X = vectorizer.transform(preprocessed_data['plot_synopsis'].values.tolist())
    # predictions = clf.predict(X.toarray()).tolist()

    # data['tags'] = ['gothic' if x == 1 else 'another' for x in predictions]
    # data['tags'].to_csv('submission.csv', sep=';')


if __name__ == "__main__":
    train()
    # prepare_submission()
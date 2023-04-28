import os

TRAIN_DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/public_train_train.csv')
TEST_DATA_PATH =  os.path.join(os.path.dirname(__file__), '../data/public_train_test.csv')

PATH_TO_VECTOR =  os.path.join(os.path.dirname(__file__), '../data/glove.6B/glove.6B.50d.txt')

SUBMISSION_DATA_TEST_PATH =  os.path.join(os.path.dirname(__file__), '../data/public_test.csv')
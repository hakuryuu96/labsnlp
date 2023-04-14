import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    data = pd.read_csv('../data/public_train.csv', index_col=0)
    
    X_train, X_test = train_test_split(data, test_size=0.2)

    X_train.to_csv('../data/public_train_train.csv')
    X_test.to_csv('../data/public_train_test.csv')
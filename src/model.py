import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import pickle as pickle
import argparse

def get_clean_data():

    data = pd.read_csv('./data/data.csv')
    df = data[['AQ', 'USS', 'CS', 'VOC', 'fail']]

    #Split Feature Variables from Target Variables
    X = df.drop('fail', axis=1)
    y = df['fail']

    return X, y

def train_eval_model(X,y, train_full_data):
    scaler = StandardScaler()
    model = LogisticRegression(random_state=14)

    if train_full_data:
        #Scale Data
        X = scaler.fit_transform(X)
        #Train model
        model.fit(X, y)

    else:
        #Split Data into test and training sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=14)

        #Scale Data
        X_train =scaler.fit_transform( X_train)
        X_test =scaler.fit_transform( X_test)

        #Train model
        model.fit(X_train, y_train)

        #Evaluate model with Cross Validation Method
        kf = KFold(n_splits=7, random_state=14, shuffle=True)
        cross_val__avg = np.mean(cross_val_score(model, X, y, cv=kf))
        print(f'The Average Cross Validation Score: {cross_val__avg}')

        #Evaluate Model w/ just train_test split 
        y_pred = model.predict(X_test)
        print(f'Model Accuracy Score with 80/20 split: {accuracy_score(y_test, y_pred)}')
        print(f'\nClassification Report:\n{classification_report(y_test, y_pred)}')

    return model, scaler


def main(train_full_data:bool=False):
    
    X, y = get_clean_data()

    model, scaler = train_eval_model(X,y, train_full_data)

    #Export Model and Scaler for Future use
    with open('./model/model.pkl', 'wb') as f:
        pickle.dump(model, f)
        
    with open('./model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model on full data or a subset.")
    parser.add_argument('--train_full_data', action='store_true', help='Flag to train on the full dataset')
    args = parser.parse_args()

    main(train_full_data=args.train_full_data)
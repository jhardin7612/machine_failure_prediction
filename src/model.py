import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import pickle as pickle

def get_clean_data():

    data = pd.read_csv('../data/data.csv')
    df = data[['AQ', 'USS', 'CS', 'VOC', 'fail']]

    return df

def create_model(data):
    #Split Feature Variables from Target Variables
    X = data.drop('fail', axis=1)
    y = data['fail']

    #Scale Data
    scaler = StandardScaler()
    X =scaler.fit_transform(X)

    #Split Data into test and training sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=14)

    #Train model
    model = LogisticRegression(random_state=14)
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


def main():
    data = get_clean_data()
    
    model, scaler = create_model(data)

    #Export Model and Scaler for Future use
    with open('../model/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('../model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)


if __name__ == '__main__':
    main()
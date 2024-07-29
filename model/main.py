import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle as pickle

def get_clean_data():

    data = pd.read_csv('../data/data.csv')

    return data

def create_model(data):
    #Split Feature Variables from Target Variables
    X = data.drop('fail', axis=1)
    y = data['fail']

    #Split Data into test and training sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=14)

    #Train model
    model = LogisticRegression(random_state=14)
    model.fit(X_train, y_train)

    #Evaluate Model 
    y_pred = model.predict(X_test)
    print(f'Model Accuracy Score: {accuracy_score(y_test, y_pred)}')
    print(f'Classification Report: {classification_report(y_test, y_pred)}')

    return model


def main():
    data = get_clean_data()
    
    model = create_model(data)

    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    main()
import streamlit as st
import pickle as pickle
import pandas as pd
import numpy as np

def get_clean_data():

    data = pd.read_csv('../data/data.csv')
    df = data[['AQ', 'USS', 'CS', 'VOC', 'fail']]

    return df

#Labels needed for metrics and sliders
labels = [
        ("Air Quality", "AQ"),
        ("Ultrasonic Sensor", "USS"), 
        ("Current Sensor", "CS"), 
        ("Volatile Organic Compound Level", "VOC"),
    ]

def create_sidebar(labels):
    st.sidebar.header("Machine Parameters")

    data = get_clean_data()

    user_input = {}
    #Loop through to create each slider and store value for later use
    for label, key in labels:
        user_input[key] = st.sidebar.slider(
            label = label,
            min_value= 0,
            max_value= data[key].max(),
            value= int(data[key].mean())
        )
    return user_input
        

def add_predictions(input_data):

    #Import and load model
    model = pickle.load(open("../model/model.pkl", "rb"))

    #Convert dictionary --> numpy array --> numpy Series
    input_array = np.array(list(input_data.values())).reshape(1,-1)
    
    #Get Prediction
    pred_results = model.predict(input_array)

    #Update Prediction Display Dynamically
    st.header("Current Prediction Status")
    if pred_results[0] == 0:
        st.success("Non-Failure")
    else:
        st.error("Fail")
    
    st.write("Probability of Failure: ", model.predict_proba(input_array)[0][1])

def main():
    st.set_page_config(
        page_title = "Machine Status Predictor",
        page_icon=":computer",
        layout="wide"
    )

    input_vals = create_sidebar(labels)

    with  st.container():
        st.title("Machine Failure Predictor")
        st.write("Manually update the parameters on the left to receive new failure prediction results.")

    st.divider()

    with st.container():
        col1, col2, col3, col4 = st.columns(4, gap="large")
        
        #Display Metrics 
        col1.metric(labels[0][0], input_vals[labels[0][1]]) #Air Quality
        col2.metric(labels[1][0], input_vals[labels[1][1]]) #Ultra Sonic
        col3.metric(labels[2][0], input_vals[labels[2][1]]) #Current Sensor
        col4.metric(labels[3][0], input_vals[labels[3][1]]) #VOC

    st.divider()

    with st.container():
        #Display Predictions
        add_predictions(input_vals)

if __name__ == '__main__':
    main()
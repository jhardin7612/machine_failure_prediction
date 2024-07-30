import streamlit as st
import pickle as pickle
import pandas as pd
import numpy as np

def get_clean_data():

    data = pd.read_csv('../data/data.csv')

    return data

#Labels needed for metrics and sliders
labels = [
        ("Footfall", "footfall"),
        ("Temperature Mode", "tempMode"), 
        ("Air Quality", "AQ"),
        ("Ultrasonic Sensor", "USS"), 
        ("Current Sensor", "CS"), 
        ("Volatile Organic Compound Level", "VOC"),
        ("Rotational Position", "RP"),
        ("Input Pressure", "IP"),
        ("Operating Temperature", "Temperature")
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
        st.write("Non-Failure")
    else:
        st.write("Fail")
    
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
        st.write("Currently this app should only be used for mission planning. Manually update individual sensor data to receive failure prediction results.")

    st.divider()

    with st.container():
        col1, col2, col3 = st.columns(3, gap="large")
        
        #Display Metrics in Grid-like pattern
        col1.metric(labels[8][0], input_vals[labels[8][1]]) #Footfall
        col1.metric(labels[1][0], input_vals[labels[1][1]]) #Temp Mode
        col1.metric(labels[2][0], input_vals[labels[2][1]]) # Air Quality

        col2.metric(labels[3][0], input_vals[labels[3][1]]) #UltraSonic
        col2.metric(labels[4][0], input_vals[labels[4][1]]) #Current Sensor
        col2.metric(labels[5][0], input_vals[labels[5][1]]) #VOC

        col3.metric(labels[6][0], input_vals[labels[6][1]]) #Rotational Position
        col3.metric(labels[7][0], input_vals[labels[7][1]]) #Input Pressure
        col3.metric(labels[0][0], input_vals[labels[0][1]]) #Temperature

    st.divider()

    with st.container():
        #Display Predictions
        add_predictions(input_vals)

if __name__ == '__main__':
    main()
import streamlit as st
import pickle as pickle
import pandas as pd


def main():
    st.set_page_config(
        page_title = "Machine Failure Predictor",
        page_icon=":computer",
        layout="wide"
    )


    with  st.container():
        st.title("Machine Failure Predictor")
        st.write("This app is designed to monitor machine sensor data in real time. Additionally, this app will allow you to manually update individual sensor data and provide a failure prediction based on the data provided.")
    
    st.divider()
    
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric('Temperature Mode', 7)
        with col2:
            st.metric('Air Quality', 7)
        with col3:
            st.metric('Ultrasonic Sensor', 1)

        


if __name__ == '__main__':
    main()
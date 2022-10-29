from operator import index
import streamlit as st
import plotly.express as px
from pycaret.regression import setup, compare_models, pull, save_model, load_model
import pandas_profiling
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os 


with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("AutoML Application")
    choice = st.radio("Navigation",["Upload","Profiling","Modelling","Download"])
    st.info("Project helps build and test automated Machine Learning Model creation Pipeline")

if os.path.exists("source_data.csv"):
    df = pd.read_csv("source_data.csv",index_col = None)

if choice == "Upload":
    st.title("Upload the Data for Modelling here !!")
    file = st.file_uploader("Upload Dataset Here")
    if file:
        df = pd.read_csv(file, index_col = None)
        df.to_csv("source_data.csv",index = None)
        st.dataframe(df)

if choice == "Profiling":
    st.title("Automated Exploratory Analysis !!")
    profile_report = df.profile_report()
    st_profile_report(profile_report)

if choice == "Modelling": 
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    if st.button('Run Modelling'): 
        setup(df, target=chosen_target, silent=True)
        setup_df = pull()
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.dataframe(compare_df)
        save_model(best_model, 'model')

if choice == "Download":
    with open("model.pkl",'rb') as f:
        st.download_button("Download Trained Model",f,file_name = 'model.pkl')
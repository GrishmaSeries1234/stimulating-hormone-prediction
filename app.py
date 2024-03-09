import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import pickle

st.set_page_config(layout="centered",page_title="Thyroid stimulating Prediction")
df=pickle.load(open("df.pkl","rb"))
model=pickle.load(open("lasso.pkl","rb"))

st.title("Thyroid stimulating prediction")
#Age
age=st.selectbox("Age",df["Age"].unique())

#Gender
gender=st.selectbox("Gender",df['Gender'].unique())

#Diseases

diseases=st.selectbox("Diseases",df['Disease'].unique())

#Biological_min
biological_min=st.selectbox("Biological_min",df["Biological_min"].unique())

#Biological_max

biological_max=st.selectbox("Biological_max",df['Biological_max'].unique())


if st.button("predict"):
      
      query_df=pd.DataFrame([[age,gender,diseases,biological_min,biological_max]],columns=['Age','Gender','Disease','Biological_min','Biological_max'])
      st.title("Therefore the predicted thyroid stimulatiing hormone:" + str(np.round(float(model.predict(query_df)[0]),3)))


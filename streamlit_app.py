import streamlit as st
import pandas as pd

# load CSV from the repo root
df = pd.read_csv("Holy Cross - Movement.csv")

st.title("HC Pitcher Max Velocities")
st.write(df.head())  # just to test
Add Streamlit app

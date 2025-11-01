import cv2
import streamlit as st
import pandas as pd
import import_ipynb
import cv2 as cv
from suggest_build import suggest_items

def recommend_items():
    if adc == None or sup == None:
        return
    else:
        items = suggest_items(adc,sup)
        with item_container:
            st.write(items)

st.title("Leona Build Suggestor")
leona_img = cv.imread("SolEcl_Leona.jpg")
leona_img = cv2.cvtColor(leona_img,cv2.COLOR_BGR2RGB)
print(type(leona_img))
st.image(leona_img)

df = pd.read_excel("LeagueChampsData.xlsx")
champs = df['Champion'].tolist()

col1, col2 = st.columns(2)
adc_champs = champs[:24]
sup_champs = champs[24:]

with col1:
    adc = st.selectbox("ADC", adc_champs, index = None)

with col2:
    sup = st.selectbox("Sup", sup_champs, index = None)
st.button("Get my build", on_click=recommend_items)
item_container = st.container()
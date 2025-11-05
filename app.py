import os
import streamlit as st
import pandas as pd
import cv2 as cv
from suggest_build_withboots import suggest_items

def recommend_items():
    item_image_location = "items"
    if adc == None or sup == None:
        return
    else:
        boots, items = suggest_items(adc,sup)
        item_list = [item[0] for item in items]
        item_images = []
        for item in item_list:
            for image in os.listdir(item_image_location):
                if item.lower() == image[:-4].lower():
                    item_image_file = cv.imread(os.path.join(item_image_location,image))
                    item_image_file = cv.cvtColor(item_image_file, cv.COLOR_BGR2RGB)
                    item_images.append(item_image_file)
        with item_container:
            st.write(boots)
            item1, item2, item3, item4, item5 = st.columns(5)
            columns = [item1, item2, item3, item4, item5]
            for i in range(len(columns)):
                with columns[i]:
                    st.image(item_images[i], width = 64)

st.title("Leona Build Suggestor")
leona_img = cv.imread("SolEcl_Leona.jpg")
leona_img = cv.cvtColor(leona_img,cv.COLOR_BGR2RGB)
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
import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import altair as alt
from PIL import Image

st.write("""
# Simple Iris Flower Prediction App

This app predicts the **Iris flower** type!

***
""")

image = Image.open('img/flowers.jpeg')

st.image(image, use_column_width=True)

st.sidebar.header('User Input Parameters')

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

data = pd.read_csv("csv/iris.csv")
def changeNum(row):
    if row == 'Iris-setosa':
        return 1
    elif row == 'Iris-versicolor':
        return 0
    else:
        return 2

data['Species_Target'] = data['Species'].apply(changeNum)

# X = data
# st.write(X)
# Y = iris.target

st.subheader('EDA(Exploratory Data Analysis)')

for i in ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']:
    f, ax = plt.subplots(figsize=(7, 5))
    # ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)
    ax = sns.swarmplot(y=i, x="Species", data=data)
    st.pyplot(f)

f, ax = plt.subplots(figsize=(7, 5))
ax =plt.scatter(data['PetalLengthCm'], data['PetalWidthCm'], c=data['Species_Target'])
plt.xlabel('Sepal Length', fontsize=18)
plt.ylabel('Sepal Width', fontsize=18)
plt.legend()
st.pyplot(f)

corrmat = data.corr()
f, ax = plt.subplots(figsize=(7, 5))
ax = sns.heatmap(corrmat, annot = True, vmax=1, square=True)
st.pyplot(f)

X = data.drop(columns=['Id','Species_Target','Species'])
Y = data.Species_Target
st.write(X)
clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(data.Species)

st.subheader('Prediction')
st.write(data.Species_Target[prediction])
#st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)




import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
from PIL import Image
import time

st.markdown("""
<style>
body {
    color: #fff;
    background-color: #90EE90;
}
</style>
    """, unsafe_allow_html=True)
st.markdown('<style>h1{color: darkgreen;}</style>', unsafe_allow_html=True)
st.write("""
# **Health Sleuth**
by Trojan Dudes
""")
image = Image.open('C:/Users/Abel Simon Zachariah/Trojan dudes/image.jpg')
st.image(image, caption = 'Trojan Dudes', use_column_width = True)
st.subheader('Please input you parameters ')


def user_input_features():
    Age = st.slider('Age', 1, 100, 50)
    Bmi = st.slider('BMI(Body Mass Index)',10 , 50, 25)
    Drinking = st.radio( 'DRINKING (NO:0, YES:1)',options = [0,1], index=1)
    Exercise = st.radio( 'EXERCISE PER WEEK',options = [1,2,3], index=1)
    Gender = st.radio('GENDER (Male:0 , Female:1)', options = [0,1])
    Junk = st.radio('JUNK FOOD CONSUMPTION PER WEEK', options = [1,2,3])
    Sleep = st.radio('SLEEP SCORE', options= [1,2,3])
    Smoking = st.radio('SMOKING (NO:0 , YES:1)', options = [0,1])
  
    data = {'Age': Age,
            'Bmi': Bmi,
            'Drinking': Drinking,
            'Exercise': Exercise,
            'Gender': Gender,
            'Junk': Junk,
            'Sleep': Sleep,
            'Smoking': Smoking,
            }
    features = pd.DataFrame(data, index=[0])
    return features

df_input = user_input_features()

st.subheader('Parameters selected by you :')
st.write(df_input)

df = pd.read_csv('dataset.csv')

chart = st.bar_chart(df)
time.sleep(1)

X = df.iloc[:, 0:8].values
y = df.iloc[:, 8:11].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler as ss
sc = ss()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


classifier= RandomForestRegressor(n_estimators = 300, random_state = 0)
classifier.fit(X_train,y_train)

prediction = classifier.predict(df_input)
st.header('Prediction')

st.spinner(text='In progress...')
ans = prediction.flatten()
a = ans[0]
b = ans[1] 
c = ans[2]

if(a<50 and b<50 and c<50):
    st.successs("You are completely fit and healthy")
elif(a>50 and a<70 and a>b and a>c):
    st.info('You have a low risk of Depression')
elif(b>50 and b<70 and b>c and b>a):
    st.info("You have a low risk of Diabetes")
elif(c>50 and c<70 and c>a and c>b):
     st.info("You have a low risk of Hypertension")           
elif(a>50 and a>b and a>c):
     st.warning("You have a high risk of Depression")         
elif (b>50 and b>a and b>c):
     st.warning("You have a high risk of Diabetes")
elif (c>50 and c>a and c>b):     
     st.warning("You have a high risk of Hypertension")
st.balloons()

prediction_proba = classifier.score(X_test,y_test)
st.subheader('Model Accuracy')
st.success(prediction_proba)

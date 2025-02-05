import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import joblib

stopwords_set=set(stopwords.words('english'))
emoji_pattern=re.compile('(?::|;|=)(?:-)?(?:\)|\(|D|P)')

tfidf=joblib.load('vectorizer.pkl')
model=joblib.load('model.pkl')

def preprocessing(txt):
    txt=re.sub('<[^>]*>','',txt)
    emojis=emoji_pattern.findall(txt)
    txt=re.sub('[\W+]',' ',txt.lower())+" ".join(emojis).replace('-','')

    prtr=PorterStemmer()
    txt=[prtr.stem(word) for word in txt.split() if word not in stopwords_set]

    return " ".join(txt)

def predict(comment):
    prepr=preprocessing(comment)
    comment_lst=[prepr]
    comment_vector=tfidf.transform(comment_lst)
    pred=model.predict(comment_vector)[0]

    ans="NOne"
    if pred==0:
        ans="Negative"
    elif pred==1:
        ans="Somewhat Negative"
    elif pred==2:
        ans="Neutral"
    elif pred==3:
        ans="Somewhat Postive"
    elif pred==4:
        ans="Positive"

    return ans

def handleMsg(msg):
    print("predicting message")
    res=predict(msg)
    print(f"{res}")
    st.session_state.res=res


if "res" not in st.session_state:
    st.session_state.res=""

st.title('Sentiment Analysis')
st.write('Enter your message we will predict it')

msg=st.text_area('The message')
st.button('Predict',handleMsg(msg))


if st.session_state.res!="":
    st.write(st.session_state.res)








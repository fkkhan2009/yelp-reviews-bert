import requests
import streamlit as st
from pydantic import BaseModel
from typing import Optional
import json


class review(BaseModel):
    text: Optional[str] = None
    stars: Optional[str] = None


st.title('Review Sentiment Analysis!!')

option = st.selectbox('How would you like to enter you review?',
                      ('Write into text', 'Speak into microphone'))

text = st.text_area('Write your review here', placeholder='Review...')



user_review = review()
print(type(user_review))
user_review.text = text
with st.spinner('Evaluating the review...'):
    res = requests.post("http://127.0.0.1:8000/predict", data=json.dumps({'text': text}))
st.success('Done!!!')
st.header('And the prediction is...')
stars = res.json()['stars']
st.subheader(f'Stars:{stars}')

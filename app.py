import streamlit as st
import pandas as pd
import numpy as np
import transformers
from transformers import pipeline
import regex as re
from emot.emo_unicode import UNICODE_EMOJI # For emojis
from emot.emo_unicode import EMOTICONS_EMO # For EMOTICONS



# Function for converting emojis into word
def convert_emojis(text):
    text = str(text)
    for emot in UNICODE_EMOJI:
        text = text.replace(emot, "_".join(UNICODE_EMOJI[emot].replace(",","").replace(":","").split()))
    return text


def clean(text):
    cleaned = re.compile(r'<.*?>')
    return re.sub(cleaned,'',str(text))

#Cleaing Special Characters
def is_special(text):
    rem = ''
    for i in text:
        if i.isalnum():
            rem = rem + i
        else:
            rem = rem + ' '
    return rem


def to_lower(text):
    return text.lower()

@st.cache
def sentiment_pipeline() -> Pipeline:
    sentiment_pipeline = pipeline("sentiment-analysis")
    return sentiment_pipeline

#Helppr Function to get sentiments
@st.cache
def get_sentiments(text):
  sentiments = sentiment_pipeline(text)
  return sentiments[0]['label']


#Dowload CSV
@st.cache
def convert_df(df):
   return df.to_csv().encode('utf-8')


st.write("""
# Text Review App
In this data app - the user upload a csv and the app display the reviews where the content doesn’t match ratings""")

st.sidebar.header('User Input Features')

# Collects user input features into dataframe
uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    #input_df = pd.read_csv(uploaded_file)
    chrome_reviews = pd.read_csv(uploaded_file)
    chrome_reviews = chrome_reviews.loc[:, ['ID','Text','Star','User Name', 'Thumbs Up','Review URL','Developer Reply', 'Version', 'Review Date', 'App ID']]
    st.write('First 3 Rows of Dataframe')
    st.write(chrome_reviews.head(3))
    st.write('Please Wait for final Result.')
    #Cleaning Text of Emojis, Speacial Characters
    chrome_reviews['Text'].dropna()
    chrome_reviews['Tokenised_Text'] = chrome_reviews['Text'].apply(convert_emojis)
    chrome_reviews['Tokenised_Text'] = chrome_reviews['Text'].apply(clean)
    chrome_reviews['Tokenised_Text'] = chrome_reviews['Tokenised_Text'].apply(is_special)
    chrome_reviews['Tokenised_Text'] = chrome_reviews['Tokenised_Text'].apply(to_lower)
    chrome_reviews['user_sentiments'] = chrome_reviews['Tokenised_Text'].apply(get_sentiments)
    chrome_reviews = chrome_reviews[chrome_reviews.user_sentiments == 'POSITIVE']
    chrome_reviews = chrome_reviews[chrome_reviews.Star < 3]
    st.subheader('Output')
    chrome_reviews[['Text','user_sentiments','Star','Tokenised_Text']]
    st.write(chrome_reviews[['Text','user_sentiments','Star','Tokenised_Text']])
    chrome_reviews = convert_df(chrome_reviews)

    st.download_button(
       "Press to Download",
       chrome_reviews,
       "file.csv",
       "text/csv",
       key='download-csv')
else:
    st.write("Please upload the CSV File with columns name as ['ID','Text','Star','User Name', 'Thumbs Up','Review URL','Developer Reply', 'Version', 'Review Date', 'App ID']")
st.subheader('Published By Lavkush Gupta')

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

#Helppr Function to get sentiments
def get_sentiments(text):
  sentiments = sentiment_pipeline(text)
  return sentiments[0]['label']


st.write("""
# Text Review App
In this data app - the user upload a csv and the app display the reviews where the content doesn’t match ratings""")

st.sidebar.header('User Input Features')

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    st.write("Please upload the file in CSV format")
#     def user_input_features():
#         pregnancies = st.sidebar.slider('No of Pregnancies',0,20,2)
#         #sex = st.sidebar.selectbox('Sex',('male','female'))
#         Glucose = st.sidebar.slider('Glucose', 0.0,400.0,120.0)
#         BloodPressure = st.sidebar.slider('BloodPressure (mm Hg)', 0.00,150.00,69.10)
#         SkinThickness = st.sidebar.slider('SkinThickness (mm)', 0.00,150.00,20.53)
#         Insulin= st.sidebar.slider('Insulin (mu U/ml)', 0.00,1000.00,79.79)
#         DiabetesPedigreeFunction = st.sidebar.slider('DiabetesPedigreeFunction (g)', -3.00,3.00,-1.01)
#         Age = st.sidebar.slider('Age (Years)', 0,110,22)
#         BMI = st.sidebar.slider('Body Mass Index ', 0.00,100.00,31.00)
#         data = {'Pregnancy': pregnancies,
#                 'Glucose': Glucose,
#                 'BloodPressure': BloodPressure,
#                 'SkinThickness': SkinThickness,
#                 'Insulin': Insulin,
#                 'BMI': BMI,
#                 'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
#                 'Age': Age,
#                 }
#         features = pd.DataFrame(data, index=[0])
#         return features
#     input_df = user_input_features()
chrome_reviews = pd.read_csv(uploaded_file)
#Cleaning Text of Emojis, Speacial Characters
chrome_reviews['Text'].dropna()
chrome_reviews['Tokenised_Text'] = chrome_reviews['Text'].apply(convert_emojis)
chrome_reviews['Tokenised_Text'] = chrome_reviews['Text'].apply(clean)
chrome_reviews['Tokenised_Text'] = chrome_reviews['Tokenised_Text'].apply(is_special)
chrome_reviews['Tokenised_Text'] = chrome_reviews['Tokenised_Text'].apply(to_lower)
chrome_reviews['user_sentiments'] = chrome_reviews['Tokenised_Text'].apply(get_sentiments)

chrome_reviews = chrome_reviews[chrome_reviews.user_sentiments == 'POSITIVE']
chrome_reviews = chrome_reviews[chrome_reviews_positive.Star < 3]

st.subheader('Output')
chrome_reviews[['Text','user_sentiments','Star','Tokenised_Text']]
st.write(chrome_reviews[['Text','user_sentiments','Star','Tokenised_Text']])



# # Combines user input features with entire Diabates dataset
# diabates_raw = pd.read_csv('https://raw.githubusercontent.com/llavkush/Diabatic-Detection/Master/diabetes.csv')
# diabates = diabates_raw.drop(columns='Outcome', axis=1, inplace = True)
# df = pd.concat([input_df,diabates],axis=0)    
    
# df = df[:1] # Selects only the first row (the user input data)
# #df1 = df.drop(columns="Outcome")
# list = df.loc[:].values.tolist() # Converting df into lists
# scaler = load(open('scaler.pkl', 'rb'))
# features = scaler.transform(list)


# # Displays the user input features
# st.subheader('User Input features')

# if uploaded_file is not None:
#     st.write(df)
# else:
#     st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
#     st.write(df)

# # Reads in saved classification model
# load_clf = joblib.load(open('model.pkl', 'rb'))

# # Apply model to make predictions
# prediction = load_clf.predict(features)
# prediction_proba = load_clf.predict_proba(features)

# st.subheader('Prediction')
# disease = np.array(['Non - Diabatic','Diabatic'])
# st.write(disease[prediction])

# st.subheader('Prediction Probability')
# st.write(prediction_proba)


st.subheader('Published By Lavkush Gupta')

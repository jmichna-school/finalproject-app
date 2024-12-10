import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression


st.markdown('# Predict LinkedIn Usage')

st.markdown('### Enter Traits Below and See Probability that a person \
            is a LinkedIn User')

st.markdown('#### App by Jake Michna')


### Train model ###

#Load in data frame of all history
s = pd.read_csv('social_media_usage.csv')

#Function checks whether x == 1, if not 1 then 0
def clean_sm(x):
    x = np.where(x == 1, 1, 0).tolist()
    return x


#Create data frame to be used for model
ss = pd.DataFrame({
    'sm_li': clean_sm(s['web1h']), 
    'income':np.where(s['income'] > 9, np.nan, s['income']), 
    'education':np.where(s['educ2'] > 8, np.nan, s['educ2']), 
    'parent':np.where(s['par'] == 1, 1, 0), 
    'married':np.where(s['marital'] == 1, 1, 0), 
    'female':np.where(s['gender'] == 2, 1, 0), 
    'age':np.where(s['age'] > 98, np.nan, s['age'])})

ss = ss.dropna()

#Create x and y variables
y = ss['sm_li']
x = ss[['income', 'education', 'parent', 'married', 'female', 'age']]

#Create logistic regression model with balance classes
lr = LogisticRegression(class_weight='balanced')

#Fit model
lr.fit(x, y)


#Sent app function
def LinkedIn_app(income, education, parent, married, female, age, model):

    #Create list from income
    x_input = [income, education, parent, married, female, age]

    #Predict class
    predicted_class = model.predict([x_input])

    #Predict probability
    probs = model.predict_proba([x_input])

    class_string = ''
    if predicted_class == 1:
        class_string = 'Yes'
    else:
        class_string = 'No'
    #Create a string object to be returned
    output_string = f'Linked In User: {class_string} --- \
    Probability person is LinkedIn User: {round(probs[0][1], 3)}'

    return output_string

#Input
income_input = st.number_input(label='Income Level (1: Low to 9: High)', \
    min_value=1, max_value=9, value=5)

education_input = st.number_input(label='Education Level (1 to 8)', \
    min_value=1, max_value=8, value=5)

parent_input = st.number_input(label='Parent (1: Yes, 0: No)', \
    min_value=0, max_value=1, value=0)


married_input = st.number_input(label='Married (1: Yes, 0: no)', \
    min_value=0, max_value=1, value=0)

female_input = st.number_input(label='Female (1: Yes, 0: No)', \
    min_value=0, max_value=1, value=0)

age_input = st.number_input(label='Age', \
    min_value=0, max_value=100, value=25)


#Model output
st.markdown(LinkedIn_app(income=income_input, education=education_input, 
                parent=parent_input, married=married_input, female=female_input, 
                age=age_input, model = lr))

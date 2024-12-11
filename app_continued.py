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
income_box = st.selectbox('Income', 
                          options = ['1. Less than $10k', 
                                     '2. $10k to under $20k', 
                                     '3. $20k to under $30k', 
                                     '4. $30k to under $40k', 
                                     '5. $40k to under $50k', 
                                     '6. $50k to under $75k', 
                                     '7. $75k to under $100k', 
                                     '8. $100k to under $150k', 
                                     '9. $150k and over'])

income_input = 0
if income_box == '1. Less than $10k':
    income_input = 1
elif income_box == '2. $10k to under $20k':
    income_input = 2
elif income_box == '3. $20k to under 30k':
    income_input = 3
elif income_box == '4. $30k to under $40k':
    income_input = 4
elif income_box == '5. $40k to under $50k':
    income_input = 5
elif income_box == '6. $50k to under $75k':
    income_input = 6
elif income_box == '7. $75k to under $100k':
    income_input = 7
elif income_box == '8. $100k to under $150k':
    income_input = 8
elif income_box == '9. $150k and over':
    income_input = 9
else:
    0

#st.write(income_input)


education_box = st.selectbox('Education Level', 
                             options = ['Less than high school', 
                                        'High school incomplete', 
                                        'High school graduate', 
                                        'Some college, no degree', 
                                        'Two-year associate degree', 
                                        'Four-year college degree', 
                                        'Some post grad schooling', 
                                        'Post grad degree'])


education_input = 0
if education_box == 'Less than high school':
    education_input = 1
elif education_box == 'High school incomplete':
    education_input = 2
elif education_box == 'High school graduate':
    education_input = 3
elif education_box == 'Some college, no degree':
    education_input = 4
elif education_box == 'Two-year associate degree':
    education_input = 5
elif education_box == 'Four-year college degree':
    education_input = 6
elif education_box == 'Some post grad schooling':
    education_input = 7
elif education_box == 'Post grad degree':
    education_input = 8
else:
    education_input = 0

#st.write(education_input)



parent_box = st.selectbox('Parent?', 
                          options=['Yes', 'No'])

parent_input = 0
if parent_box == 'Yes':
    parent_input = 1
else:
    parent_input = 0

#st.write(parent_input)


married_box = st.selectbox('Married?', 
                           options=['Yes', 'No'])

married_input = 0
if married_box == 'Yes':
    married_input = 1
else:
    married_input = 0


#st.write(married_input)


female_box = st.selectbox('Female?', 
                          options=['Yes', 'No'])

female_input = 0
if female_box == 'Yes':
    female_input = 1
else:
    female_input = 0

#st.write(female_input)

age_input = st.number_input(label='Age', \
    min_value=0, max_value=100, value=25)


#Model output
st.markdown(LinkedIn_app(income=income_input, education=education_input, 
                parent=parent_input, married=married_input, female=female_input, 
                age=age_input, model = lr))

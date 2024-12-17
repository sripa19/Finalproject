import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import streamlit as st

# Load and preprocess data
s = pd.read_csv('social_media_usage.csv')
ss = s[['income', 'educ2', 'web1i', 'par', 'marital', 'age']]  # Replace 'web1h' with 'web1i' for Instagram usage
ss = ss[~((ss['income'] > 9) | 
         (ss['educ2'] > 8) |
         (ss['web1i'] > 2) |  # Filter for valid Instagram responses
         (ss['par'] > 2) | 
         (ss['marital'] > 6) |
         (ss['age'] > 98) |
          ss.isna().any(axis=1))]
ss.rename(columns={'web1i': 'sm_instagram'}, inplace=True)
y = ss['sm_instagram']
X = ss[['income', 'educ2', 'par', 'marital', 'age']]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X.values,
                                                    y,
                                                    stratify=y,       # Same number of target in training & test set
                                                    test_size=0.2,    # Hold out 20% of data for testing
                                                    random_state=987)

# Train logistic regression model
lr = LogisticRegression(class_weight='balanced', random_state=987)
lr.fit(X_train, y_train)

# Streamlit app
st.title("Instagram User Prediction")

st.markdown("### Data Definitions:")
st.markdown("- **Age**: 1 to 97")
st.markdown("- **Education**: 1 Less than high school, 2 High school incomplete, 3 High school graduate, 4 Some college no degree, 5 Two-year associate degree, 6 Four-year college, 7 Some postgraduate or professional schooling, 8 Postgraduate or professional degree")
st.markdown("- **Parent**: 1 Yes, 2 No")
st.markdown("- **Current Marital Status**: 1 Married, 2 Living with a partner, 3 Divorced, 4 Separated, 5 Widowed, 6 Never been married")
st.markdown("- **Income**: 1 (less money), 9 (lots of money)")

# User input fields
age = st.number_input("Age", min_value=1, max_value=97, step=1)
education = st.number_input("Education", min_value=1, max_value=8, step=1)
parent = st.number_input("Parent", min_value=1, max_value=2, step=1)
marital = st.number_input("Marital Status", min_value=1, max_value=6, step=1)
income = st.number_input("Income", min_value=1, max_value=9, step=1)

# Prediction
if st.button("Predict Instagram User"):
    # Create input array for prediction
    user_data = np.array([[income, education, parent, marital, age]])
    
    # Make prediction and compute probability
    prediction = lr.predict(user_data)
    probability = lr.predict_proba(user_data)[:, 1]  # Probability for class 1 (Instagram user)
    
    # Display the results
    if prediction == 1:
        st.success(f"The person is predicted to be an Instagram user with a probability of {probability[0]:.2f}.")
    else:
        st.info(f"The person is predicted to NOT be an Instagram user with a probability of {probability[0]:.2f}.")





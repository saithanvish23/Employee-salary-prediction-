import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# --- Page Configuration ---
st.set_page_config(page_title="Employee Salary Predictor", page_icon="💼", layout="centered")

# --- Load and Prepare Data ---
@st.cache_data
def load_data():
    np.random.seed(42)
    job_titles = ['Software Engineer', 'Data Scientist', 'Product Manager', 'HR Executive', 'Marketing Specialist']
    education_levels = ['High School', 'Bachelors', 'Masters', 'PhD']
    departments = ['IT', 'HR', 'Marketing', 'Finance', 'Operations']
    locations = ['New York', 'San Francisco', 'Chicago', 'Austin', 'Remote']
    work_types = ['Remote', 'In-office', 'Hybrid']

    data = {
        'Job Title': np.random.choice(job_titles, 500),
        'Education Level': np.random.choice(education_levels, 500),
        'Department': np.random.choice(departments, 500),
        'Location': np.random.choice(locations, 500),
        'Work Type': np.random.choice(work_types, 500),
        'Years of Experience': np.random.randint(0, 31, 500),
    }

    df = pd.DataFrame(data)
    base_salary = 40000
    df['Salary'] = base_salary + \
                   df['Years of Experience'] * 1000 + \
                   df['Job Title'].apply(lambda x: job_titles.index(x) * 5000) + \
                   np.random.randint(0, 20000, 500)
    return df

df = load_data()

# --- Label Encode Categorical Features ---
def encode_features(df, encoders=None):
    if encoders is None:
        encoders = {}
        for col in ['Job Title', 'Education Level', 'Department', 'Location', 'Work Type']:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
    else:
        for col in ['Job Title', 'Education Level', 'Department', 'Location', 'Work Type']:
            df[col] = encoders[col].transform(df[col])
    return df, encoders

df_encoded, label_encoders = encode_features(df.copy())

# --- Train Gradient Boosting Regressor Model ---
X = df_encoded.drop("Salary", axis=1)
y = df_encoded["Salary"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, max_depth=4, random_state=42)
model.fit(X_train, y_train)

# --- Streamlit Title & Description ---
st.title("💼 Employee Salary Prediction")
st.markdown("Predict employee salaries based on job profile, education, and work preferences using **Gradient Boosting**.")

# --- Sidebar Info ---
with st.sidebar:
    st.header(" Model Information")
    st.markdown("""
    - **Model**: Gradient Boosting Regressor  
    - **Records**: 500 (synthetic)  
    - **Features**: Job, Education, Dept., Location, Work Type, Experience  
    """)
    st.success("Model trained within this app!")

# --- Input Form ---
with st.form("prediction_form"):
    st.subheader(" Enter Employee Details")

    col1, col2 = st.columns(2)
    with col1:
        job = st.selectbox("Job Title", label_encoders['Job Title'].classes_)
        edu = st.selectbox("Education Level", label_encoders['Education Level'].classes_)
        dept = st.selectbox("Department", label_encoders['Department'].classes_)

    with col2:
        loc = st.selectbox("Location", label_encoders['Location'].classes_)
        work = st.selectbox("Work Type", label_encoders['Work Type'].classes_)
        exp = st.slider("Years of Experience", 0, 30, 3)

    submitted = st.form_submit_button("Predict Salary 💰")

# --- Make Prediction ---
if submitted:
    input_data = pd.DataFrame({
        'Job Title': [job],
        'Education Level': [edu],
        'Department': [dept],
        'Location': [loc],
        'Work Type': [work],
        'Years of Experience': [exp]
    })

    for col in input_data.columns:
        input_data[col] = label_encoders[col].transform(input_data[col]) if col in label_encoders else input_data[col]

    prediction = model.predict(input_data)[0]
    st.success(f" **Estimated Salary:** ${prediction:,.2f}")

# --- Salary Distribution ---
with st.expander(" Show Salary Distribution"):
    fig, ax = plt.subplots()
    sns.histplot(df["Salary"], kde=True, bins=30, ax=ax, color='lightblue')
    if submitted:
        ax.axvline(prediction, color='red', linestyle='--', label="Predicted Salary")
        ax.legend()
    ax.set_title("Salary Distribution")
    st.pyplot(fig)



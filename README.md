# Employee-salary-prediction-
:

💼 Employee Salary Prediction
This project is a Streamlit-based web app that predicts employee salaries based on various input features such as job title, education level, department, location, work type, and years of experience.
The app uses a Gradient Boosting Regressor trained on synthetic data to estimate salaries.

🚀 Features
✅ Interactive UI built with Streamlit
✅ Synthetic dataset generation (no external CSV required)
✅ Automatic encoding of categorical features
✅ Machine Learning model: Gradient Boosting Regressor
✅ Visualizes salary distribution with predicted value marked
✅ Runs completely locally in your browser

📂 Project Structure
bash
Copy
Edit
Employee_Salary_Prediction.py   # Main Streamlit app
README.md                       # Project documentation
⚙️ Installation
Clone this repository

bash
Copy
Edit
git clone https://github.com/your-username/employee-salary-prediction.git
cd employee-salary-prediction
Set up a virtual environment (recommended)

bash
Copy
Edit
python -m venv venv
source venv/bin/activate      # On Linux/Mac
venv\Scripts\activate         # On Windows
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
(If you don’t have a requirements.txt, you can create one:)

txt
Copy
Edit
streamlit
pandas
numpy
scikit-learn
seaborn
matplotlib
▶️ Running the App
Run the Streamlit app:

bash
Copy
Edit
streamlit run Employee_Salary_Prediction.py
Then open the local URL shown in your terminal (usually http://localhost:8501).

🧪 How It Works
The app generates a synthetic dataset of 500 employees with random attributes.

Categorical features are label-encoded.

A Gradient Boosting Regressor is trained on this dataset.

Users input employee details in the sidebar form.

The app predicts the estimated salary and visualizes the distribution.

📊 Example Features
Job Title: Software Engineer, Data Scientist, etc.

Education Level: High School, Bachelors, Masters, PhD

Department: IT, HR, Marketing, etc.

Location: New York, San Francisco, etc.

Work Type: Remote, In-office, Hybrid

Years of Experience: 0–30

💡 Future Enhancements
Use real-world datasets

Add more ML models for comparison

Save user input history

Deploy on Streamlit Cloud or Hugging Face Spaces

📜 License
This project is open-source and available under the MIT License.


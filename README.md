Iris Flower Classification – ML Web Application


This project is a machine learning-powered web app that predicts the species of iris flowers using physical measurements (sepal/petal dimensions). It features:

A Streamlit-based visual interface for model training, evaluation, and interactive predictions.

A Flask-based user authentication portal for secure prediction access with login/register functionalities.

Built using the classic Iris dataset, this app demonstrates how ML models can be embedded into modern web interfaces.

Requirements – Libraries & Dependencies
Make sure the following libraries are installed in your local Python environment:


pip install streamlit flask pandas seaborn matplotlib scikit-learn
If you prefer, use a requirements.txt:


streamlit
flask
pandas
seaborn
matplotlib
scikit-learn





Full Application Flow – End to End
1. Dataset Loading (iris.csv)
A standard dataset with 150 samples of iris flowers.

Columns: sepal_length, sepal_width, petal_length, petal_width, species.

2. Model Preparation (Shared by Both Apps)
Features (X) and target (y) are separated.

Data split into training and test sets using train_test_split.

Two classifiers are trained:

RandomForestClassifier

LogisticRegression


Streamlit Interface: streamlit_app.py


🔹 Features:
Dataset preview (first 10 rows).
<img width="1738" height="867" alt="image" src="https://github.com/user-attachments/assets/65e52d2e-ed92-45fb-adf1-6429218b9ba3" />

Missing values and duplicate checks.
<img width="1226" height="791" alt="image" src="https://github.com/user-attachments/assets/d9232f07-7863-4149-9314-d1b5ff7c5f20" />
<img width="1245" height="632" alt="image" src="https://github.com/user-attachments/assets/5d1f025c-3dc2-4886-a7ec-891b6ddb6c49" />


Trains both ML models and shows:

Accuracy score

Classification reports
<img width="1099" height="587" alt="image" src="https://github.com/user-attachments/assets/c33017ce-b70f-42c2-8403-e93304d7029c" />

Confusion matrices (heatmaps)
<img width="1214" height="607" alt="image" src="https://github.com/user-attachments/assets/6c135429-f5c9-4cf4-9a9a-19405b9471be" />


Accuracy comparison bar chart
<img width="1231" height="813" alt="image" src="https://github.com/user-attachments/assets/83b3b3c9-f468-47f4-b85f-9db2f7ea4ad4" />


Multiclass ROC curves
<img width="1056" height="804" alt="image" src="https://github.com/user-attachments/assets/20991b50-9d05-44a7-88fd-385c4685c93c" />


Interactive prediction form with input sliders.

Option to choose which model to use.

Random forest model:
<img width="1157" height="837" alt="image" src="https://github.com/user-attachments/assets/3a89243a-1a39-4248-9ce9-d6da77bcca35" />

Logistic regression model:
<img width="1147" height="846" alt="image" src="https://github.com/user-attachments/assets/ae19bb8e-e9c8-42c7-998e-5db00366026d" />



Clean UI with headings for clarity.

▶️ Run Streamlit:


streamlit run streamlit_app.py






Flask Interface: app.py


🔹 Features:
User registration and login system using SQLite.
<img width="1900" height="884" alt="Screenshot 2025-07-28 150209" src="https://github.com/user-attachments/assets/0c3312f3-8534-4225-ac95-9d5c8f8559f2" />
<img width="1900" height="884" alt="Screenshot 2025-07-28 150209" src="https://github.com/user-attachments/assets/89d323b8-e25e-4218-8254-e7905ab689b2" />

After login, user is redirected to a prediction form.
<img width="1899" height="886" alt="Screenshot 2025-07-28 150308" src="https://github.com/user-attachments/assets/756824ee-839e-48b8-8b32-f8d7365e6155" />


Users input flower measurements and choose the model.
Random forest model:
<img width="1898" height="878" alt="Screenshot 2025-07-28 150526" src="https://github.com/user-attachments/assets/53514d2d-e9f2-4dc3-8304-e25749a8ed3e" />
<img width="1905" height="882" alt="Screenshot 2025-07-28 150551" src="https://github.com/user-attachments/assets/0a603360-b6a8-4a91-9283-f9cea3892ec4" />

Logistic Regression model:
<img width="1895" height="877" alt="Screenshot 2025-07-28 150741" src="https://github.com/user-attachments/assets/3243e7de-e4cc-4396-bcef-0350b5e508f1" />
<img width="1910" height="881" alt="Screenshot 2025-07-28 150758" src="https://github.com/user-attachments/assets/af9123be-7a43-406c-b8b0-5a85137fc0bc" />



Prediction result is shown directly on the page.

Basic session management (logout, user flash messages).

Key Routes:
Route	Functionality
/	Redirects to login
/register	New user registration
/login	Login with credentials
/predict	Prediction form + result
/logout	Session logout

▶️ Run Flask:

python app.py


Database users.db will be created automatically.

Extra Features
Dual interfaces for flexibility:

Streamlit: visual, ideal for demos & exploration.

Flask: structured, login-secured user flow.

Clean UI/UX with Matplotlib + Seaborn charts.

Multiclass ROC and bar chart comparisons.

Login state maintained using Flask sessions.

SQLite DB integration with validation and error handling.

Future Scope:
This project is just a demo – a working proof of concept for deploying ML models in web apps.

I’m currently working on a production-grade module that includes:

Full security with hashed passwords

User roles and access control

Docker containerisation

REST APIs with authentication

Admin dashboard and analytics

CI/CD for scalable deployment

This app is just a small part of a bigger system being developed.

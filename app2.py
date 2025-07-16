from flask import Flask, render_template, request, redirect, url_for, session, flash
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import os

app = Flask(__name__)
app.secret_key = 'secret'

# Load dataset and train models
iris_df = pd.read_csv('iris.csv')
x = iris_df.drop('species', axis=1)
y = iris_df['species']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)

lr_model = LogisticRegression(max_iter=200)
lr_model.fit(x_train, y_train)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        sl = float(request.form['sepal_length'])
        sw = float(request.form['sepal_width'])
        pl = float(request.form['petal_length'])
        pw = float(request.form['petal_width'])
        model = request.form['model']

        input_df = pd.DataFrame([[sl, sw, pl, pw]], columns=x.columns)
        if model == 'rf':
            prediction = rf_model.predict(input_df)[0]
        else:
            prediction = lr_model.predict(input_df)[0]

    return render_template('predict.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

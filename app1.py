from flask import Flask, render_template, request, redirect, url_for, flash, session
import sqlite3, os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import io
import base64

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# ---------- Database ----------
def init_db():
    if not os.path.exists('users.db'):
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('''
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fullname TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        ''')
        conn.commit()
        conn.close()

# ---------- Model Preparation ----------
df = pd.read_csv('iris.csv')
X = df.drop('species', axis=1)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

lr_model = LogisticRegression(max_iter=200)
lr_model.fit(X_train, y_train)

# ---------- Routes ----------
@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        fullname = request.form['fullname']
        email = request.form['email']
        password = request.form['password']

        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        try:
            c.execute('INSERT INTO users (fullname, email, password) VALUES (?, ?, ?)', (fullname, email, password))
            conn.commit()
            flash("Registered successfully. Please login.", "success")
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash("Email already registered.", "danger")
        finally:
            conn.close()
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE email = ? AND password = ?', (email, password))
        user = c.fetchone()
        conn.close()

        if user:
            session['user'] = user[1]  # fullname
            return redirect(url_for('predict'))
        else:
            flash("Invalid email or password", "danger")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    flash("Logged out successfully.", "info")
    return redirect(url_for('login'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user' not in session:
        return redirect(url_for('login'))

    prediction_result = None
    if request.method == 'POST':
        sl = float(request.form['sl'])
        sw = float(request.form['sw'])
        pl = float(request.form['pl'])
        pw = float(request.form['pw'])
        model_choice = request.form['model']

        input_data = pd.DataFrame([[sl, sw, pl, pw]], columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

        if model_choice == 'Random Forest':
            prediction_result = rf_model.predict(input_data)[0]
        else:
            prediction_result = lr_model.predict(input_data)[0]

    return render_template('predict.html', user=session['user'], prediction=prediction_result)

if __name__ == '__main__':
    init_db()
    app.run(debug=True)

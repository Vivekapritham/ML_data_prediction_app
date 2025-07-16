import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

# Load Dataset
st.title("🌸 Iris Flower Classification App")
df = pd.read_csv('iris.csv')
st.subheader("🔍 First 10 Rows of the Dataset")
st.dataframe(df.head(10))

# Missing and Duplicate Check
st.subheader("🧹 Data Preprocessing")
st.write("Missing Value Check:")
st.dataframe(df.head(10).isnull())

st.write("Missing Value Count:")
st.dataframe(df.head(10).isnull().sum())

st.write("Duplicate Rows in First 10:")
duplicates = df.head(10)[df.head(10).duplicated()]
st.dataframe(duplicates)

# Split features and target
x = df.drop('species', axis=1)
y = df['species']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(x_train, y_train)
rf_pred = rf.predict(x_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_report = classification_report(y_test, rf_pred)
rf_conf_matrix = confusion_matrix(y_test, rf_pred)

# Train Logistic Regression
lr = LogisticRegression(max_iter=200)
lr.fit(x_train, y_train)
lr_pred = lr.predict(x_test)
lr_accuracy = accuracy_score(y_test, lr_pred)
lr_report = classification_report(y_test, lr_pred)
lr_conf_matrix = confusion_matrix(y_test, lr_pred)

# Results Section
st.subheader("📈 Model Results")

col1, col2 = st.columns(2)

with col1:
    st.write("**Random Forest Accuracy:**", rf_accuracy)
    st.text("Classification Report:")
    st.text(rf_report)

with col2:
    st.write("**Logistic Regression Accuracy:**", lr_accuracy)
    st.text("Classification Report:")
    st.text(lr_report)

# Confusion Matrices
st.subheader("🔷 Confusion Matrices")

col1, col2 = st.columns(2)
with col1:
    st.write("Random Forest")
    fig, ax = plt.subplots()
    sns.heatmap(rf_conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=rf.classes_, yticklabels=rf.classes_, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    st.pyplot(fig)

with col2:
    st.write("Logistic Regression")
    fig, ax = plt.subplots()
    sns.heatmap(lr_conf_matrix, annot=True, fmt='d', cmap='Greens', 
                xticklabels=lr.classes_, yticklabels=lr.classes_, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    st.pyplot(fig)

# Accuracy Comparison Charts
st.subheader("📊 Accuracy Comparison")
fig, ax = plt.subplots()
sns.barplot(x=["Random Forest", "Logistic Regression"], y=[rf_accuracy, lr_accuracy], palette='viridis', ax=ax)

for i, acc in enumerate([rf_accuracy, lr_accuracy]):
    ax.text(i, acc + 0.01, f"{acc:.2f}", ha='center')
ax.set_ylim(0, 1.05)
ax.set_ylabel("Accuracy")
st.pyplot(fig)

# ROC Curve
st.subheader("🧪 ROC Curve Comparison")

y_test_bin = label_binarize(y_test, classes=rf.classes_)
n_classes = y_test_bin.shape[1]

# ROC for Random Forest
rf_probs = rf.predict_proba(x_test)
fpr_rf, tpr_rf, roc_auc_rf = {}, {}, {}
for i in range(n_classes):
    fpr_rf[i], tpr_rf[i], _ = roc_curve(y_test_bin[:, i], rf_probs[:, i])
    roc_auc_rf[i] = auc(fpr_rf[i], tpr_rf[i])

# ROC for Logistic Regression
lr_probs = lr.predict_proba(x_test)
fpr_lr, tpr_lr, roc_auc_lr = {}, {}, {}
for i in range(n_classes):
    fpr_lr[i], tpr_lr[i], _ = roc_curve(y_test_bin[:, i], lr_probs[:, i])
    roc_auc_lr[i] = auc(fpr_lr[i], tpr_lr[i])

# Plot ROC
fig, ax = plt.subplots()
colors = ['blue', 'green', 'red']
for i in range(n_classes):
    ax.plot(fpr_rf[i], tpr_rf[i], linestyle='--', label=f'RF Class {rf.classes_[i]} AUC={roc_auc_rf[i]:.2f}', color=colors[i])
    ax.plot(fpr_lr[i], tpr_lr[i], linestyle='-', label=f'LR Class {lr.classes_[i]} AUC={roc_auc_lr[i]:.2f}', color=colors[i])
ax.plot([0, 1], [0, 1], 'k--')
ax.set_title("ROC Curve")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend()
st.pyplot(fig)

# Prediction
st.subheader("🔮 Predict Flower Species")
with st.form("prediction_form"):
    sl = st.number_input("Sepal Length (cm)", min_value=0.0, value=5.1)
    sw = st.number_input("Sepal Width (cm)", min_value=0.0, value=3.5)
    pl = st.number_input("Petal Length (cm)", min_value=0.0, value=1.4)
    pw = st.number_input("Petal Width (cm)", min_value=0.0, value=0.2)
    model_choice = st.selectbox("Choose Model", ["Random Forest", "Logistic Regression"])
    submitted = st.form_submit_button("Predict")

if submitted:
    user_input = pd.DataFrame([[sl, sw, pl, pw]], 
                              columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    if model_choice == "Random Forest":
        prediction = rf.predict(user_input)[0]
    else:
        prediction = lr.predict(user_input)[0]

    st.success(f"🌼 Predicted Species: **{prediction}**")

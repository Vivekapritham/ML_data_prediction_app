import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
#data loading
df=pd.read_csv('iris.csv')
print(df.head(10))
#data pre processing
print("missing value matrix (True= missing):")
print(df.head(10).isnull())
#handling missing values
print("missing value count:")
print(df.head(10).isnull().sum())
#handling duplicate values
print("checking duplicates present or not:")
print(df.head(10).duplicated())
print("duplicate rows:")
print(df.head(10)[df.head(10).duplicated()])
#data splitting
x=df.drop('species', axis=1) #features
y=df['species']  #target
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2, random_state=42)
print("x_train shape:",x_train.shape)
print("x_test shape:", x_test.shape)
print("y_train shape:",y_train.shape)
print("y_test shape:", y_test.shape)
#train random forest
print("...Random Forest...")
rf=RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(x_train,y_train)
rf_pred=rf.predict(x_test)
rf_accuracy=accuracy_score(y_test,rf_pred)
rf_report=classification_report(y_test, rf_pred)
rf_conf_matrix=confusion_matrix(y_test,rf_pred)
print("Predictions:",rf_pred)
print("Accuracy:",rf_accuracy)
print("Classification Report:",rf_report)
print("Confusion matrix:",rf_conf_matrix)
print("True Labels:",y_test.values)
#plot confusion matrix
plt.figure(figsize=(4, 3))
sns.heatmap(rf_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=rf.classes_, yticklabels=rf.classes_)
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.show()
#train logistic regression
print("...Logistic Regression...")
lr=LogisticRegression(max_iter=200)
lr.fit(x_train,y_train)
lr_pred=lr.predict(x_test)
lr_accuracy=accuracy_score(y_test,lr_pred)
lr_report=classification_report(y_test,lr_pred)
lr_conf_matrix=confusion_matrix(y_test,lr_pred)
print("Predictions:",lr_pred)
print("Accuracy:",lr_accuracy)
print("Classification Report:",lr_report)
print("Confusion matrix:",lr_conf_matrix)
plt.figure(figsize=(4, 3))
sns.heatmap(lr_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=rf.classes_, yticklabels=rf.classes_)
plt.title('Logistic Regression Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.show()
#plotting accuracy comparison bar chart
plt.figure(figsize=(5,4))
models=['Random Forest','Logistic Regression']
accuracies=[rf_accuracy, lr_accuracy]
sns.barplot(x=models, y=accuracies, palette='viridis')
plt.title('Accuracy Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim(0,1.05)
for i,acc in enumerate(accuracies):
    plt.text(i,acc+0.2, f"{acc:.2f}",ha='center')
plt.tight_layout()
plt.show() 

#plotting roc curve
# Binarize the output for ROC Curve
y_test_bin = label_binarize(y_test, classes=rf.classes_)
n_classes = y_test_bin.shape[1]

# ROC for Random Forest
rf_probs = rf.predict_proba(x_test)
fpr_rf = dict()
tpr_rf = dict()
roc_auc_rf = dict()
for i in range(n_classes):
    fpr_rf[i], tpr_rf[i], _ = roc_curve(y_test_bin[:, i], rf_probs[:, i])
    roc_auc_rf[i] = auc(fpr_rf[i], tpr_rf[i])

# ROC for Logistic Regression
lr_probs = lr.predict_proba(x_test)
fpr_lr = dict()
tpr_lr = dict()
roc_auc_lr = dict()
for i in range(n_classes):
    fpr_lr[i], tpr_lr[i], _ = roc_curve(y_test_bin[:, i], lr_probs[:, i])
    roc_auc_lr[i] = auc(fpr_lr[i], tpr_lr[i])

# Plotting ROC curves
plt.figure(figsize=(8, 6))
colors = ['blue', 'green', 'red']
for i in range(n_classes):
    plt.plot(fpr_rf[i], tpr_rf[i], linestyle='--', label=f'RF ROC curve (class {rf.classes_[i]}) AUC = {roc_auc_rf[i]:.2f}', color=colors[i])
    plt.plot(fpr_lr[i], tpr_lr[i], linestyle='-', label=f'LR ROC curve (class {rf.classes_[i]}) AUC = {roc_auc_lr[i]:.2f}', color=colors[i])

plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison (Random Forest vs Logistic Regression)')
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()
#prediction
try:
    print("\nEnter flower features to predict its species:")

    sepal_length = float(input("Enter sepal length (cm): "))
    sepal_width = float(input("Enter sepal width (cm): "))
    petal_length = float(input("Enter petal length (cm): "))
    petal_width = float(input("Enter petal width (cm): "))

    # Create a DataFrame for prediction
    user_input_df = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                                 columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

    # Logistic Regression prediction
    pred_lr = lr.predict(user_input_df)[0]

    # Random Forest prediction
    pred_rf = rf.predict(user_input_df)[0]

    print(f"\n Logistic Regression Prediction: {pred_lr}")
    print(f" Random Forest Prediction: {pred_rf}")

except Exception as e:
    print(" Error during user input prediction:",e)

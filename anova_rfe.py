import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import matplotlib.pyplot as plt
import seaborn as sns

# อ่านข้อมูล
waterdata = pd.read_csv('water_potability.csv')

# แยก Features และ Target
X = waterdata.drop(columns='Potability')
X.fillna(X.mean(), inplace=True)  # แทนที่ missing values ด้วยค่าเฉลี่ย
Y = waterdata['Potability']

# แบ่งข้อมูลเป็นชุด train และ test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# ---------- SMOTE เพื่อเพิ่มข้อมูลในคลาสที่น้อย ----------
smote = SMOTE()
X_train_smote, Y_train_smote = smote.fit_resample(X_train, Y_train)

# ---------- Standardization ----------
scaler = StandardScaler()
X_train_smote_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)

# ---------- Logistic Regression (Baseline) ----------
waterLogis = LogisticRegression()
waterLogis.fit(X_train_smote_scaled, Y_train_smote)
Logispred = waterLogis.predict(X_test_scaled)

print("Logistic Regression Accuracy (Baseline):", accuracy_score(Y_test, Logispred))
cm = confusion_matrix(Y_test, Logispred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='g', cbar=False, xticklabels=['Not Potable', 'Potable'], yticklabels=['Not Potable', 'Potable'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - Logistic Regression (Baseline)')
plt.show()

# ---------- ANOVA (SelectKBest) ----------
anova_selector = SelectKBest(f_classif, k=5)  # เลือก 5 ฟีเจอร์
X_train_anova = anova_selector.fit_transform(X_train_smote_scaled, Y_train_smote)
X_test_anova = anova_selector.transform(X_test_scaled)

# ฝึก Logistic Regression ด้วยฟีเจอร์ที่เลือกจาก ANOVA
logreg_anova = LogisticRegression()
logreg_anova.fit(X_train_anova, Y_train_smote)
anova_pred = logreg_anova.predict(X_test_anova)

print("\nLogistic Regression Accuracy (ANOVA):", accuracy_score(Y_test, anova_pred))
print("Classification Report (ANOVA):\n", classification_report(Y_test, anova_pred))

# Confusion Matrix สำหรับ ANOVA
cm_anova = confusion_matrix(Y_test, anova_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm_anova, annot=True, fmt='g', cbar=False, xticklabels=['Not Potable', 'Potable'], yticklabels=['Not Potable', 'Potable'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - Logistic Regression (ANOVA)')
plt.show()

# ---------- RFE (Recursive Feature Elimination) ----------
logreg_rfe = LogisticRegression()
rfe_selector = RFE(logreg_rfe, n_features_to_select=5)  # เลือก 5 ฟีเจอร์
X_train_rfe = rfe_selector.fit_transform(X_train_smote_scaled, Y_train_smote)
X_test_rfe = rfe_selector.transform(X_test_scaled)

# ฝึก Logistic Regression ด้วยฟีเจอร์ที่เลือกจาก RFE
logreg_rfe.fit(X_train_rfe, Y_train_smote)
rfe_pred = logreg_rfe.predict(X_test_rfe)

print("\nLogistic Regression Accuracy (RFE):", accuracy_score(Y_test, rfe_pred))
print("Classification Report (RFE):\n", classification_report(Y_test, rfe_pred))

# Confusion Matrix สำหรับ RFE
cm_rfe = confusion_matrix(Y_test, rfe_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm_rfe, annot=True, fmt='g', cbar=False, xticklabels=['Not Potable', 'Potable'], yticklabels=['Not Potable', 'Potable'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - Logistic Regression (RFE)')
plt.show()

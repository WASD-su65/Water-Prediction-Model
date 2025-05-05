import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('water_potability.csv')

print(df.isnull().sum())
df.fillna(df.mean(), inplace=True)
print(df.isnull().sum())
print(df.info())
print("Shape ของข้อมูล",df.shape)
print(df.dtypes)

x = df.drop('Potability', axis=1)  # Features เอาคอลัมน์ Potability ออก
y = df['Potability']                     # Target คือคอลัมน์ Potability
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

LGT_model = LogisticRegression()
LGT_model.fit(x_train, y_train)

y_pred = LGT_model.predict(x_test)

print("Logistic Accuracy : ", accuracy_score(y_test, y_pred))
print("Classification Report : ", classification_report(y_test, y_pred))
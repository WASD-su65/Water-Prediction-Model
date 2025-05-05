import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


df = pd.read_csv('water_potability.csv')

print(df.isnull().sum())

df.fillna(df.median(), inplace=True)
print(df.isnull().sum())

print(df.info())
print("Shape ของข้อมูล",df.shape)

print(df.dtypes)

X = df.drop(columns='Potability')
y = df['Potability']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

RFC = RandomForestClassifier(n_estimators=300, random_state=42)
RFC.fit(X_train, y_train)

y_pred = RFC.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
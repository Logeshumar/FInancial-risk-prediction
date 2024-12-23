import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabeleNCODER
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
train_df = pd.read_csv('dataset\Train.csv')
test_df = pd.read_csv('dataset\Test.csv')
print('Train dataset shape:,', train_df.shape)
print('Test dataset shape:,', test_df.shape)
train_df.head()
train_df.into()
train_df.describe()
missing_values = train_df.isnull().sum()
print('Missing values:\n',missing_values)
sns.countplot(data=train_df, x='IsUnderRisk')
plt.title('Distribution of IsUnderRisk')
plt.show()
plt.figure(figsize=(10, 8))
sns.hetmap(train_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
sns.boxplot(data=traion_df, x='IsUnderRisk', y='Location_Score')
plt.title('Location Score by Risk Status')
plt.show()
selected_features = ['Location_Score', 'Internal_Audit_Score', 'External_Audit_Score', 'Fin_Score']
sns.pairplot(train_df[selected_features + ['IsUnderRisk']], hue='IsUnderRisk')
plt.show()
if 'City' in train_df.columns:
  le + LabelEncoder()
  train_df['City'] = le.fit_transform(train_df['City'])
  test_df['City'] = test_df['City'].map(lambda s: le.classes_.tolist().index(s) if s in le.classes_ else -1)
X = train_df.drop('IsUnderRisk', axis=1)
y = train_df['IsUnderRisk']
scaler = StandardSclar()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test_df)
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
clf = RandomForestClassifier(random_state=42)
clf.flt(X_train, y_train)
y_pred = clf.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print('Validation Accuracy:', accuracy)
print('Classification Report:\n', classification_report(y_val, y_pred))
conf_matrix = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='YlGnBu', fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
test_predictions = clf.predict(test_scaled)
test_df['IsUnderRisk_Prediction'] = test_predeictions
test_df.head()


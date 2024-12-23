import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load datasets
train_df = pd.read_csv('dataset/Train.csv')
test_df = pd.read_csv('dataset/Test.csv')
print('Train dataset shape:', train_df.shape)
print('Test dataset shape:', test_df.shape)

# Inspect datasets
print(train_df.head())
print(train_df.info())
print(train_df.describe())

# Check for missing values
missing_values = train_df.isnull().sum()
print('Missing values:\n', missing_values)

# Visualize the distribution of 'IsUnderRisk'
sns.countplot(data=train_df, x='IsUnderRisk')
plt.title('Distribution of IsUnderRisk')
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(train_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Boxplot for 'Location_Score' by 'IsUnderRisk'
sns.boxplot(data=train_df, x='IsUnderRisk', y='Location_Score')
plt.title('Location Score by Risk Status')
plt.show()

# Pairplot of selected features
selected_features = ['Location_Score', 'Internal_Audit_Score', 'External_Audit_Score', 'Fin_Score']
sns.pairplot(train_df[selected_features + ['IsUnderRisk']], hue='IsUnderRisk')
plt.show()

# Encode categorical variables
if 'City' in train_df.columns:
    le = LabelEncoder()
    train_df['City'] = le.fit_transform(train_df['City'])
    test_df['City'] = test_df['City'].map(lambda s: le.classes_.tolist().index(s) if s in le.classes_ else -1)

# Feature selection and scaling
X = train_df.drop('IsUnderRisk', axis=1)
y = train_df['IsUnderRisk']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test_df)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Model training
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Validation
y_pred = clf.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print('Validation Accuracy:', accuracy)
print('Classification Report:\n', classification_report(y_val, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='YlGnBu', fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Predictions on test data
test_predictions = clf.predict(test_scaled)
test_df['IsUnderRisk_Prediction'] = test_predictions
print(test_df.head())

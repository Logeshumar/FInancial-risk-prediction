1. Import Libraries
The code begins by importing libraries required for:
Data manipulation: pandas and numpy.
Visualization: matplotlib.pyplot and seaborn.
Machine learning tasks: sklearn modules for preprocessing, model training, evaluation, and hyperparameter tuning.
2. Load Datasets
python
Copy code
train_df = pd.read_csv('dataset/Train.csv')
test_df = pd.read_csv('dataset/Test.csv')
The training dataset (Train.csv) is used to train the model.
The test dataset (Test.csv) is used to make predictions on unseen data.
The shapes of both datasets are printed for initial inspection.
3. Inspect and Explore the Data
Preview Data:

python
Copy code
print(train_df.head())
Displays the first five rows of the training dataset to understand its structure and contents.

Dataset Info:

python
Copy code
print(train_df.info())
Provides a summary of column names, data types, and non-null values, helping identify missing data or incorrect data types.

Statistical Summary:

python
Copy code
print(train_df.describe())
Displays descriptive statistics for numerical columns, such as mean, median, and standard deviation.

4. Check for Missing Values
python
Copy code
missing_values = train_df.isnull().sum()
print('Missing values:\n', missing_values)
Identifies and counts missing values in each column, allowing for further preprocessing steps like filling or removing missing data.
5. Visualize Target Variable
python
Copy code
sns.countplot(data=train_df, x='IsUnderRisk')
plt.title('Distribution of IsUnderRisk')
plt.show()
Creates a bar plot to visualize the class distribution of the target variable IsUnderRisk.
Helps determine if the dataset is imbalanced (i.e., one class significantly outweighs the other).
6. Correlation Heatmap
python
Copy code
plt.figure(figsize=(10, 8))
sns.heatmap(train_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
Computes correlations between numerical features using train_df.corr().
Displays a heatmap to identify:
Strongly correlated features (useful for predictive models).
Redundant features (potentially removable).
7. Boxplot Analysis
python
Copy code
sns.boxplot(data=train_df, x='IsUnderRisk', y='Location_Score')
plt.title('Location Score by Risk Status')
plt.show()
Displays how the Location_Score feature varies for each class in IsUnderRisk.
Helps identify whether Location_Score is a distinguishing factor for predicting risk.
8. Pairplot for Feature Relationships
python
Copy code
selected_features = ['Location_Score', 'Internal_Audit_Score', 'External_Audit_Score', 'Fin_Score']
sns.pairplot(train_df[selected_features + ['IsUnderRisk']], hue='IsUnderRisk')
plt.show()
Visualizes pairwise relationships between selected numerical features.
Highlights patterns or clusters based on the IsUnderRisk target variable.
9. Encode Categorical Variables
python
Copy code
if 'City' in train_df.columns:
    le = LabelEncoder()
    train_df['City'] = le.fit_transform(train_df['City'])
    test_df['City'] = test_df['City'].map(lambda s: le.classes_.tolist().index(s) if s in le.classes_ else -1)
Encodes the City column (categorical feature) into numerical values using LabelEncoder.
Ensures consistency between training and test datasets by mapping unseen categories to -1.
10. Feature Scaling
python
Copy code
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test_df)
Scales numerical features to standardize the data.
Improves model performance by normalizing feature ranges.
11. Split Data into Training and Validation Sets
python
Copy code
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
Splits the training data into:
X_train, y_train: For training the model.
X_val, y_val: For validating model performance.
A 70-30 split is used.
12. Train the Model
python
Copy code
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
Trains a RandomForestClassifier using the training data.
13. Validate the Model
python
Copy code
y_pred = clf.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print('Validation Accuracy:', accuracy)
print('Classification Report:\n', classification_report(y_val, y_pred))
Predicts target values for validation data.
Evaluates model performance using:
Accuracy Score: Measures overall correctness.
Classification Report: Provides precision, recall, and F1-score for each class.
14. Display Confusion Matrix
python
Copy code
conf_matrix = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='YlGnBu', fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
Creates a confusion matrix to visualize:
True positives, true negatives, false positives, and false negatives.
Helps assess prediction errors.
15. Make Predictions on Test Data
python
Copy code
test_predictions = clf.predict(test_scaled)
test_df['IsUnderRisk_Prediction'] = test_predictions
print(test_df.head())
Uses the trained model to predict the target variable for test data.
Adds predictions as a new column (IsUnderRisk_Prediction) to the test dataset.
Prints the first few rows of the updated test dataset.


import pandas as pd
import numpy as np

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree

# Machine Learning: Preprocessing, Models, and Metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

data = {
    'Timestamp': pd.to_datetime(pd.date_range(start='2023-01-01', periods=200, freq='H')),
    'Location': np.random.choice(['Downtown', 'Suburb', 'Highway', 'Arterial'], 200),
    'Vehicle_Count': np.random.randint(10, 101, 200),
    'Vehicle_Speed': np.random.randint(10, 71, 200),
    'Peak_Off_Peak': np.random.choice(['Peak', 'Off-Peak'], 200),
    'Congestion_Level': np.random.randint(0, 6, 200),
    'Target_Vehicle_Count': np.random.randint(10, 101, 200) # This is the leaky feature
}
df_dummy = pd.DataFrame(data)

# To make the relationships more realistic (as found in EDA)
# We'll enforce some correlations
df_dummy['Congestion_Level'] = (df_dummy['Vehicle_Count'] // 20) + ( (70 - df_dummy['Vehicle_Speed']) // 15)
df_dummy['Congestion_Level'] = df_dummy['Congestion_Level'].clip(0, 5).astype(int)

# Save to a CSV file to simulate a real-world file loading scenario
csv_filename = 'urban_traffic_flow.csv'
df_dummy.to_csv(csv_filename, index=False)

# --- Start of the main analysis ---
print("--- Loading Data ---")
df = pd.read_csv(csv_filename)
print(f"Successfully loaded '{csv_filename}' with {df.shape[0]} rows and {df.shape[1]} columns.\n")


# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ----------------------------------
print("--- 1. Exploratory Data Analysis (EDA) ---")

# Initial data inspection
print("\nFirst 5 rows of the dataset:")
print(df.head())

print("\nDataset information (columns, data types, null values):")
df.info()

print("\nDescriptive statistics for numerical columns:")
print(df.describe())

# Visualize distributions
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Distribution of Key Features', fontsize=16)

sns.histplot(df['Vehicle_Count'], kde=True, ax=axes[0, 0], color='skyblue')
axes[0, 0].set_title('Vehicle Count Distribution')

sns.histplot(df['Vehicle_Speed'], kde=True, ax=axes[0, 1], color='salmon')
axes[0, 1].set_title('Vehicle Speed Distribution')

sns.countplot(x='Congestion_Level', data=df, ax=axes[1, 0], palette='viridis')
axes[1, 0].set_title('Congestion Level Counts')

sns.countplot(x='Peak_Off_Peak', data=df, ax=axes[1, 1], palette='crest')
axes[1, 1].set_title('Peak vs. Off-Peak Counts')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Correlation analysis
print("\nCorrelation matrix of numerical features:")
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr(numeric_only=True)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()
print("EDA complete. Key relationships visualized.\n")


# 4. DATA PREPROCESSING AND FEATURE ENGINEERING
# ---------------------------------------------
print("--- 2. Data Preprocessing and Feature Engineering ---")

# Convert Timestamp to datetime object and extract hour
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['hour'] = df['Timestamp'].dt.hour

# Create the binary target variable: 'is_high_congestion'
# We define 'High Congestion' as a level of 3 or more.
df['is_high_congestion'] = (df['Congestion_Level'] >= 3).astype(int)
print("Created binary target variable 'is_high_congestion'.")
print(df['is_high_congestion'].value_counts(normalize=True))

# Define features (X) and target (y)
# IMPORTANT: We drop 'Target_Vehicle_Count' to prevent data leakage.
# We also drop 'Congestion_Level' (it's our source for the target) and 'Timestamp'.
features = ['Vehicle_Count', 'Vehicle_Speed', 'hour', 'Location', 'Peak_Off_Peak']
target = 'is_high_congestion'

X = df[features]
y = df[target]

print(f"\nFeatures (X) selected: {X.columns.tolist()}")
print(f"Target (y) selected: {target}")

# Split data into training and testing sets
# We use stratify=y to ensure the proportion of classes is the same in train and test sets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"\nData split into training ({X_train.shape[0]} samples) and testing ({X_test.shape[0]} samples) sets.")


# 5. MODELING PIPELINE SETUP
# --------------------------
# Define which columns are numerical and which are categorical
numerical_features = ['Vehicle_Count', 'Vehicle_Speed', 'hour']
categorical_features = ['Location', 'Peak_Off_Peak']

# Create a preprocessor object using ColumnTransformer.
# This applies different transformations to different columns.
# - 'passthrough' leaves the numerical columns as they are.
# - OneHotEncoder converts categorical columns into numerical format.
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough' # Keep other columns if any (safer)
)
print("\nPreprocessor created to handle categorical features.")


# 6. LOGISTIC REGRESSION MODEL
# ----------------------------
print("\n--- 3. Training and Evaluating Logistic Regression Model ---")

# Create the full pipeline with preprocessing and the model
log_reg_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

# Train the model
log_reg_pipeline.fit(X_train, y_train)
print("Logistic Regression model trained successfully.")

# Make predictions on the test set
y_pred_log_reg = log_reg_pipeline.predict(X_test)

# Evaluate the model
print("\nClassification Report for Logistic Regression:")
print(classification_report(y_test, y_pred_log_reg))

# Visualize the Confusion Matrix
cm_log_reg = confusion_matrix(y_test, y_pred_log_reg)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_log_reg, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not High', 'High'], yticklabels=['Not High', 'High'])
plt.title('Confusion Matrix: Logistic Regression')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Interpret the model coefficients
print("\nInterpreting Logistic Regression Coefficients:")
# Get feature names after one-hot encoding from the trained pipeline
ohe_feature_names = log_reg_pipeline.named_steps['preprocessor'] \
                                    .named_transformers_['cat'] \
                                    .get_feature_names_out(categorical_features)
all_feature_names = numerical_features + list(ohe_feature_names)

# Get the coefficients from the trained model
coefficients = log_reg_pipeline.named_steps['classifier'].coef_[0]

# Create a DataFrame for better visualization
coef_df = pd.DataFrame({'Feature': all_feature_names, 'Coefficient': coefficients})
coef_df = coef_df.sort_values('Coefficient', ascending=False)
print(coef_df)
print("Coefficients show the influence of each feature on the prediction.")


# 7. DECISION TREE MODEL
# ----------------------
print("\n--- 4. Training and Evaluating Decision Tree Model ---")

# Create the Decision Tree pipeline
tree_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(max_depth=5, random_state=42))
])

# Train the model
tree_pipeline.fit(X_train, y_train)
print("Decision Tree model trained successfully.")

# Make predictions
y_pred_tree = tree_pipeline.predict(X_test)

# Evaluate the model
print("\nClassification Report for Decision Tree:")
print(classification_report(y_test, y_pred_tree))

# Visualize the Confusion Matrix
cm_tree = confusion_matrix(y_test, y_pred_tree)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_tree, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Not High', 'High'], yticklabels=['Not High', 'High'])
plt.title('Confusion Matrix: Decision Tree')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Visualize the Decision Tree for ultimate interpretability
print("\nVisualizing the learned Decision Tree:")
plt.figure(figsize=(25, 15))
plot_tree(tree_pipeline.named_steps['classifier'],
          feature_names=all_feature_names,
          class_names=['Not High', 'High'],
          filled=True,
          rounded=True,
          fontsize=10)
plt.title("Decision Tree for Traffic Congestion Prediction", fontsize=20)
plt.show()
print("\n--- End of Script ---")

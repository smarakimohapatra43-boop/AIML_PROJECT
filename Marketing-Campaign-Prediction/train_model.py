import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

print("Loading data...")
df = pd.read_csv('c:/Sakshi/bank.csv')

print("Preprocessing data...")
categorical_cols_df = df.select_dtypes(include='object').columns.drop('deposit', errors='ignore')

# Apply one-hot encoding
X_encoded = pd.get_dummies(df.drop('deposit', axis=1), columns=categorical_cols_df, drop_first=True)

le_full = LabelEncoder()
y_encoded = le_full.fit_transform(df['deposit'])

print("Training model...")
model_rf = RandomForestClassifier(criterion='entropy', max_depth=8, max_features='log2', random_state=42)
model_rf.fit(X_encoded, y_encoded)

print("Saving model and features...")
# Save the model
joblib.dump(model_rf, 'c:/Sakshi/random_forest_model.pkl')

# Save the column names so we can recreate the same shape in Streamlit
joblib.dump(list(X_encoded.columns), 'c:/Sakshi/model_features.pkl')

print("Done! Model and features saved.")

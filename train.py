import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import joblib

# 1. Load your climate dataset
try:
    df = pd.read_csv("carbon_emission_data.csv")
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: 'carbon_emission_data.csv' not found.")
    exit()

# 2. Define your exact target and feature columns
target_column = '2011'
categorical_cols = ['Country code', 'Country name', 'Series code', 'Series name']
numerical_cols = ['SCALE', 'Decimals', '1990', '1995', '2000', '2005', '2010']

print("Running ultra-stable cleaning layer...")

# Force categorical columns to clean strings
for col in categorical_cols:
    df[col] = df[col].astype(str).fillna("Unknown")

# Force numerical features to numbers (coercing strings/errors to 0)
for col in numerical_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

# Force target column to numbers
df[target_column] = pd.to_numeric(df[target_column], errors='coerce')

# Drop records where the final target variable is missing
df = df.dropna(subset=[target_column])
print(f"Data ready! Total clean rows for training: {len(df)}")

# Separate features and target
X = df[categorical_cols + numerical_cols]
y = df[target_column]

# 3. Upgraded Preprocessing Layer
# Using OrdinalEncoder prevents matrix explosions from hundreds of country strings
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_cols)
    ],
    remainder='drop'
)

# 4. Assemble the Pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
])

# 5. Split and Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training your Custom Climate AI Model pipeline...")
model_pipeline.fit(X_train, y_train)

# 6. Evaluate and Save
score = model_pipeline.score(X_test, y_test)
print(f"🎉 Success! Model Training Complete.")
print(f"📈 R^2 Predictive Accuracy Score: {score:.4f}")

joblib.dump(model_pipeline, "climate_model_pipeline.pkl")
print("Saved production asset as 'climate_model_pipeline.pkl'!")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error
import joblib

DATA_PATH = "Train.csv"
df = pd.read_csv(DATA_PATH)

target_col = 'SQUARE_FT'
X = df.drop(columns=[target_col, 'ADDRESS'])
y = df[target_col]

num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training the model...")
model_pipeline.fit(X_train, y_train)

preds = model_pipeline.predict(X_val)
rmse = root_mean_squared_error(y_val, preds)
print(f"Validation RMSE: {rmse:.4f}")

joblib.dump(model_pipeline, 'house_price_model.pkl')
print("Model saved successfully as 'house_price_model.pkl'")
# train_model.py
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
import pickle

# 1. Load your enriched sales data
# Ensure this filename matches your Excel file in the Explorer pane
df = pd.read_excel("226842_196b62e8465_127512.xlsx")

# 2. Feature engineering
# Extract sale month and compute vehicle age
df['sale_month'] = pd.to_datetime(df['Sold Date']).dt.month.astype(str)
df['age'] = pd.to_datetime(df['Sold Date']).dt.year - df['Year']

# 3. Select features and target
feature_cols = [
    'Make', 'Model', 'Series', 'Mileage', 'Grade',
    'Engine Code', 'Drivable', 'Auction Region',
    'Color', 'Roof', 'Interior', 'sale_month', 'age'
]
# Drop rows with missing critical values
df = df.dropna(subset=feature_cols + ['Sale Price'])

# Split features vs. target
X = df[feature_cols].copy()
# Cast categorical fields to string for OneHotEncoder
num_feats = ['Mileage', 'Grade', 'age']
cat_feats = [c for c in feature_cols if c not in num_feats]
X[cat_feats] = X[cat_feats].astype(str)
y = np.log1p(df['Sale Price'])  # log1p transform

# 4. Build preprocessing & modeling pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_feats),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_feats),
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', HistGradientBoostingRegressor(random_state=42))
])

# 5. Train/test split and fit
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
pipeline.fit(X_train, y_train)

# 6. Serialize pipeline for Streamlit
pickle_path = 'model_pipeline.pkl'
with open(pickle_path, 'wb') as f:
    pickle.dump(pipeline, f)
print(f"✅ {pickle_path} written")
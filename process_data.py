import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import pickle

print("Starting data processing and model training...")

# --- STEP 1: LOAD AND COMBINE DATA ---
try:
    delhi_df = pd.read_csv('Indian_housing_Delhi_data.csv')
    mumbai_df = pd.read_csv('Indian_housing_Mumbai_data.csv')
    pune_df = pd.read_csv('Indian_housing_Pune_data.csv')
except FileNotFoundError:
    print("Error: Make sure all three CSV files are in the same folder as this script.")
    exit()

delhi_df['city'] = 'Delhi'
mumbai_df['city'] = 'Mumbai'
pune_df['city'] = 'Pune'

df = pd.concat([delhi_df, mumbai_df, pune_df], ignore_index=True)
print("Data loaded and combined.")

# --- STEP 2: CLEAN AND PREPROCESS DATA ---
columns_to_drop = ['isNegotiable', 'priceSqFt', 'description', 'currency', 'verificationDate']
df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

# --- NEW, CRUCIAL FIX: Clean the location column ---
# This removes all hidden leading/trailing whitespace (e.g., " Andheri East " -> "Andheri East")
df['location'] = df['location'].str.strip()
print("Cleaned whitespace from location names.")


df['house_size'] = df['house_size'].astype(str).str.extract(r'(\d+)').astype(float)
df['price'] = df['price'].astype(str).str.replace(',', '').astype(float)
df['SecurityDeposit'] = df['SecurityDeposit'].astype(str).str.replace(',', '').str.replace('No Deposit', '0')
df['SecurityDeposit'] = pd.to_numeric(df['SecurityDeposit'], errors='coerce')

df['numBathrooms'] = df['numBathrooms'].fillna(df['numBathrooms'].median())
df['numBalconies'] = df['numBalconies'].fillna(df['numBalconies'].median())
df['SecurityDeposit'] = df['SecurityDeposit'].fillna(df['SecurityDeposit'].median())

df['bhk'] = df['house_type'].str.extract(r'(\d+)').astype(int)
print("Data cleaning complete.")

# --- STEP 3: PREPARE DATA FOR MODELING ---
features_to_use = ['city', 'location', 'house_size', 'bhk', 'numBathrooms', 'Status']
target_variable = 'price'

final_df = df[features_to_use + [target_variable]].dropna()

encoders = {}
for col in ['city', 'location', 'Status']:
    le = LabelEncoder()
    final_df[col] = le.fit_transform(final_df[col])
    encoders[col] = le

X = final_df[features_to_use]
y = final_df[target_variable]
print("Data prepared for modeling.")

# --- STEP 4: TRAIN THE RANDOM FOREST MODEL ---
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1) # Use all CPU cores
model.fit(X, y)
print("Model training complete.")

# --- STEP 5: SAVE THE ARTIFACTS ---
with open('rf_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)

df.to_csv('df_cleaned_for_app.csv', index=False)
print("Successfully saved 'rf_model.pkl', 'encoders.pkl', and 'df_cleaned_for_app.csv'.")
print("\n--- SCRIPT FINISHED ---")


import pandas as pd
import numpy as np
import joblib 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')

def load_and_clean_data():
    """Loads, combines, and performs cleaning (including house_size fix) on the data."""
    print("Starting data processing and model training...")
    
    # Load individual city datasets
    try:
        delhi_df = pd.read_csv('Indian_housing_Delhi_data.csv')
        mumbai_df = pd.read_csv('Indian_housing_Mumbai_data.csv')
        pune_df = pd.read_csv('Indian_housing_Pune_data.csv')
    except FileNotFoundError:
        print("Error: Make sure all three CSV files are present.")
        raise

    # Add city column and combine
    delhi_df['city'] = 'Delhi'
    mumbai_df['city'] = 'Mumbai'
    pune_df['city'] = 'Pune'
    df = pd.concat([delhi_df, mumbai_df, pune_df], ignore_index=True)
    
    # Initial Cleaning
    # latitude and longitude are RETAINED for spatial modeling.
    columns_to_drop = ['isNegotiable', 'priceSqFt', 'description', 'currency', 'verificationDate']
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    # 1. House Size: Remove commas (FIX), extract numeric part, and convert to float
    df['house_size'] = (df['house_size'].astype(str)
                        .str.replace(',', '', regex=False)
                        .str.extract(r'(\d+.\d+|\d+)')
                        .astype(float))

    # 2. Price: Remove commas, extract numeric part, and convert to float
    df['price'] = (df['price'].astype(str)
                   .str.replace(',', '', regex=False)
                   .str.extract(r'(\d+)')
                   .astype(float))

    # 3. SecurityDeposit: Handle 'No Deposit' as 0, remove commas, and convert to float
    df['SecurityDeposit'] = (df['SecurityDeposit'].astype(str)
                             .str.replace(',', '', regex=False)
                             .str.replace('No Deposit', '0')
                             .str.extract(r'(\d+)')
                             .astype(float))
    
    # 4. BHK: Extract number of bedrooms from 'house_type'
    df['bhk'] = df['house_type'].str.extract(r'(\d+)').astype(float)
    df.drop(columns=['house_type'], inplace=True, errors='ignore') # Drop the raw house_type column

    # Clean whitespace
    for col in ['location', 'Status']:
        df[col] = df[col].astype(str).str.strip()
    
    # Impute NaNs with median (now includes spatial features)
    num_cols = ['price', 'house_size', 'numBathrooms', 'numBalconies', 'SecurityDeposit', 'bhk', 'latitude', 'longitude']
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
        
    df.dropna(subset=['location', 'Status'], inplace=True)

    # --- CRITICAL FIX: house_size Outlier Capping ---
    Q1 = df['house_size'].quantile(0.25)
    Q3 = df['house_size'].quantile(0.75)
    IQR = Q3 - Q1
    UPPER_BOUND = Q3 + 3.0 * IQR
    MIN_SIZE = 100.0

    df['house_size'] = np.where(df['house_size'] < MIN_SIZE, MIN_SIZE, df['house_size'])
    df['house_size'] = np.where(df['house_size'] > UPPER_BOUND, UPPER_BOUND, df['house_size'])
    
    print("Data cleaning and house_size correction complete.")
    return df

def train_city_models(df):
    """Trains and saves city-specific Random Forest models."""
    
    # Feature sets for the spatially aware model
    NUMERICAL_FEATURES = ['house_size', 'numBathrooms', 'numBalconies', 'SecurityDeposit', 'bhk', 'latitude', 'longitude']
    CATEGORICAL_FEATURES = ['location', 'Status']
    CITIES = df['city'].unique()
    
    # Preprocessing Pipeline Definition (Global)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERICAL_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES)
        ],
        remainder='drop' 
    )
    
    print(f"\nTraining separate models for: {CITIES}")
    
    rf_model_def = RandomForestRegressor(n_estimators=150, max_depth=20, random_state=42, n_jobs=-1)

    for city in CITIES:
        print(f"\n--- Training Model for {city} ---")
        
        df_city = df[df['city'] == city].copy()
        
        # Define X and y 
        X = df_city[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
        y = np.log1p(df_city['price']) # Log transformation on target

        # Split Data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Define and Train the Pipeline (fit for this specific city's data)
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', rf_model_def)
        ]).fit(X_train, y_train)

        # Evaluate the Model
        y_pred_test = model.predict(X_test)
        y_test_exp = np.expm1(y_test)
        y_pred_exp = np.expm1(y_pred_test)

        r2 = r2_score(y_test_exp, y_pred_exp)
        mae = mean_absolute_error(y_test_exp, y_pred_exp)
        
        print(f"Test R-squared: {r2:.4f}")
        print(f"Test MAE (Mean Absolute Error): ₹ {mae:,.0f}")
        
        # Save the Model
        model_filename = f'rf_model_{city.lower()}.pkl'
        joblib.dump(model, model_filename)
        print(f"Model saved as: {model_filename}")

if __name__ == '__main__':
    # 1. Load, Clean, and Correct Data
    df_cleaned = load_and_clean_data()
    
    # 2. Train Models
    train_city_models(df_cleaned)
    
    # 3. Save small data file for app dropdowns (includes coordinates)
    df_cleaned[['city', 'location', 'Status', 'latitude', 'longitude']].drop_duplicates().to_csv('df_city_locations_and_coords.csv', index=False)
    print("\nSaved 'df_city_locations_and_coords.csv' for Streamlit app dropdowns.")
    print("All tasks complete. You can now run 'streamlit run app.py'")

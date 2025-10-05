# Indian Rental Price Predictor

A machine learning web application that predicts monthly rental prices for properties in Delhi, Mumbai, and Pune using Random Forest regression.

## Features

- Predict rental prices for properties in major Indian cities
- Interactive web interface built with Streamlit
- City-based location filtering
- Property customization options (size, bedrooms, bathrooms, furnishing status)
- Clean, professional UI with blue color scheme
- Responsive design with wide sidebar for better usability

## Dataset

The application uses housing data from three major Indian cities:
- Delhi (5,000 records, 288 locations)
- Mumbai (5,000 records, 189 locations) 
- Pune (3,910 records, 225 locations)

Total dataset: 13,910 property records across 701 unique locations.

## Requirements

### Python Version
- Python 3.7 or higher

### Required Libraries
```
pandas>=1.4.0
scikit-learn>=1.0.0
streamlit>=1.0.0
pickle (built-in)
```

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/indian-rental-price-predictor.git
   cd indian-rental-price-predictor
   ```

2. **Install required packages**
   ```bash
   pip install pandas scikit-learn streamlit
   ```

   Or if you prefer using pip3:
   ```bash
   pip3 install pandas scikit-learn streamlit
   ```

## Setup and Usage

### Step 1: Process the Data and Train the Model

Before running the web application, you need to process the raw data and train the machine learning model:

```bash
python process_data.py
```

This script will:
- Load and combine data from Delhi, Mumbai, and Pune CSV files
- Clean and preprocess the data
- Train a Random Forest regression model
- Save the trained model (`rf_model.pkl`) and encoders (`encoders.pkl`)
- Generate a cleaned dataset (`df_cleaned_for_app.csv`)

### Step 2: Run the Web Application

After processing the data, start the Streamlit web application:

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## How to Use the Application

1. **Select City**: Choose from Delhi, Mumbai, or Pune
2. **Select Location**: Pick a specific location within the selected city
3. **Configure Property Details**:
   - House Size: Set the property size in square feet (200-8000 sq ft)
   - Bedrooms (BHK): Choose number of bedrooms (1-10)
   - Bathrooms: Select number of bathrooms (1-10)
   - Furnishing Status: Choose from available furnishing options
4. **Get Prediction**: Click "Predict Rent" to see the estimated monthly rental price

## File Structure

```
indian-rental-price-predictor/
├── app.py                          # Streamlit web application
├── process_data.py                 # Data processing and model training script
├── script.py                       # Script to view Mumbai locations
├── debug_locations.py              # Debug script for data analysis
├── test_app_logic.py              # Testing script for app functionality
├── Indian_housing_Delhi_data.csv   # Raw Delhi housing data
├── Indian_housing_Mumbai_data.csv  # Raw Mumbai housing data
├── Indian_housing_Pune_data.csv    # Raw Pune housing data
├── df_cleaned_for_app.csv          # Processed dataset (generated)
├── rf_model.pkl                    # Trained Random Forest model (generated)
├── encoders.pkl                    # Label encoders for categorical data (generated)
└── README.md                       # This file
```

## Model Details

- **Algorithm**: Random Forest Regression
- **Features**: City, Location, House Size, BHK, Number of Bathrooms, Furnishing Status
- **Target**: Monthly Rental Price (in INR)
- **Preprocessing**: Label encoding for categorical variables, data cleaning and validation

## Data Processing Features

- Automatic whitespace cleaning for location names
- Missing value imputation using median values
- Currency and numeric formatting standardization
- Feature extraction from text fields (BHK from house type)


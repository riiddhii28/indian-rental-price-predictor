Indian Rental Price Predictor: City-Specific Models

A machine learning web application that predicts monthly rental prices for properties in Delhi, Mumbai, and Pune using three separate, location-specific Random Forest regression models for enhanced accuracy.

Project Files

ds_lab_proj_new.ipynb: The data science pipeline (Cleaning, EDA, Training).

app.py: The Streamlit web application.

requirements.txt: List of Python dependencies.

Indian_housing_...data.csv (3 files): Your raw input data.

Generated Files: rf_model_delhi.pkl, rf_model_mumbai.pkl, rf_model_pune.pkl, and df_city_locations.csv (created after running the notebook).

Setup and Usage

Step 1: Install Dependencies

Make sure you have all required libraries installed:

pip install -r requirements.txt


Step 2: Process Data and Train Models (MANDATORY)

You must run the Jupyter notebook first to clean the data (fixing the suspicious house_size correlation) and train the three localized models.

Open ds_lab_proj_new.ipynb (e.g., in VS Code or Jupyter).

Run all cells sequentially.

This step verifies the corrected correlation heatmap, shows the accuracy metrics for each city, and saves the three necessary model files and the location data file.

Step 3: Run the Web Application

After processing the data in the notebook, start the Streamlit web application:

streamlit run app.py


The application will open in your web browser.
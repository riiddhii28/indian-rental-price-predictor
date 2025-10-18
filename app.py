import streamlit as st
import pandas as pd
import joblib
import numpy as np
import time

# --- CUSTOM CSS FOR AESTHETICS (Dark Blue/Black Scheme) ---
# This CSS targets Streamlit's default components to apply the requested color scheme.
# Primary Color (Dark Blue)
PRIMARY_COLOR = "#004080" 
# Secondary Color (Slightly Lighter Blue for Sliders/Buttons)
SECONDARY_COLOR = "#0066cc" 
# Text Color (Black)
TEXT_COLOR = "#111111" 

custom_css = f"""
<style>
    /* 1. Global Background/Text Color */
    .stApp {{
        color: {TEXT_COLOR};
    }}
    
    /* 2. Customizing Sliders (Blue instead of Streamlit default Red) */
    div.stSlider > div[data-testid='stTrack'] {{
        background: {SECONDARY_COLOR}50; /* 50 is opacity */
    }}
    div.stSlider > div[data-testid='stTrack'] > div[data-testid='stFill'] {{
        background: {SECONDARY_COLOR};
    }}
    
    /* 3. Customizing Submit Button (Predict Rent) */
    .stButton > button {{
        background-color: {PRIMARY_COLOR}; /* Dark blue background */
        color: white; /* White text for contrast */
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: bold;
        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2);
        transition: all 0.2s ease-in-out;
    }}
    .stButton > button:hover {{
        background-color: {SECONDARY_COLOR}; /* Lighter blue on hover */
        color: white;
    }}

    /* 4. Sidebar Header Color */
    .st-emotion-cache-1pxxpyj h1 {{
        color: {PRIMARY_COLOR};
    }}
    
    /* 5. General Text Readability */
    h1, h2, h3, h4, .stMarkdown, .stSelectbox label, .stSlider label, .stNumberInput label {{
        color: {TEXT_COLOR};
    }}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


# --- CONFIGURATION ---
st.set_page_config(
    page_title="City-Specific Rental Price Predictor (Spatial)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LOAD ASSETS (Data & Models) ---

@st.cache_resource
def load_models_and_data():
    """Loads all city models and the location data (including coordinates)."""
    model_files = {
        'Delhi': 'rf_model_delhi.pkl',
        'Mumbai': 'rf_model_mumbai.pkl',
        'Pune': 'rf_model_pune.pkl'
    }
    
    # Load Models
    models = {}
    for city, file in model_files.items():
        try:
            models[city] = joblib.load(file)
        except FileNotFoundError:
            st.error(f"Model file not found for {city}: {file}. Please run process_data.py first.")
            return None, None
            
    # Load Data for Dropdowns and Coordinates
    try:
        df_locations = pd.read_csv('df_city_locations_and_coords.csv')
    except FileNotFoundError:
        st.error("Data file 'df_city_locations_and_coords.csv' not found. Please run process_data.py first.")
        return None, None
        
    return models, df_locations

models, df_locations = load_models_and_data()

if models is None or df_locations is None:
    st.stop()


# --- PREDICTION FUNCTION ---

def make_prediction(model, input_df):
    """
    Makes a prediction using the selected model. The input must include 
    all 9 features expected by the pipeline.
    """
    
    # Feature list must match the order used during model training
    expected_features = ['house_size', 'numBathrooms', 'numBalconies', 'SecurityDeposit', 'bhk', 'latitude', 'longitude', 'location', 'Status']
    
    input_df = input_df[expected_features]
    
    # Predict returns log(price)
    log_predicted_price = model.predict(input_df)[0]
    
    # Inverse transform (expm1) to get the final price in INR
    predicted_price = np.expm1(log_predicted_price)
    
    return predicted_price


# --- STREAMLIT UI ---

st.title("₹ Your Local Price Meter")
st.markdown("""
Your local, coordinate-verified rental value for Delhi, Mumbai, and Pune.
""")

# --- SIDEBAR (INPUTS) ---

with st.sidebar:
    st.header("Property Details")
    CITY_OPTIONS = sorted(list(models.keys()))
    selected_city = st.selectbox(
        "1. Select City:",
        options=[None] + CITY_OPTIONS,
        index=0,
        format_func=lambda x: "--- Select a City ---" if x is None else x,
        key='city_selector'
    )
    inputs_enabled = selected_city is not None

    df_city_filtered = pd.DataFrame()
    city_locations = []
    selected_location = None
    if inputs_enabled:
        df_city_filtered = df_locations[df_locations['city'] == selected_city].copy()
        city_locations = sorted(df_city_filtered['location'].unique())
        selected_location = st.selectbox(
            "2. Select Specific Location/Area:",
            options=city_locations,
            key='location_selector',
            index=0
        )
    else:
        st.selectbox(
            "2. Select Specific Location/Area:",
            options=['Please select a city first'],
            disabled=True
        )

    st.markdown("---")

    house_size = st.slider(
        "3. House Size (Sq. Ft.):",
        min_value=200, max_value=4000, value=900, step=50, disabled=not inputs_enabled
    )

    # Instead of columns, place each slider one below the other
    bhk = st.slider(
        "4. Bedrooms (BHK):",
        min_value=1, max_value=6, value=2, disabled=not inputs_enabled
    )

    num_bathrooms = st.slider(
        "5. Bathrooms:",
        min_value=1, max_value=6, value=2, disabled=not inputs_enabled
    )

    num_balconies = st.slider(
        "6. Balconies:",
        min_value=0, max_value=5, value=1, disabled=not inputs_enabled
    )

    security_deposit = st.number_input(
        "7. Security Deposit (INR):",
        min_value=0, value=50000, step=10000, disabled=not inputs_enabled
    )

    status_options = sorted(df_locations['Status'].unique())
    default_status_index = status_options.index('Furnished') if 'Furnished' in status_options else 0

    selected_status = st.selectbox(
        "8. Furnishing Status:",
        options=status_options,
        index=default_status_index,
        disabled=not inputs_enabled
    )

    st.markdown("---")

    predict_enabled = selected_city is not None and selected_location is not None
    submit_button = st.button(label='Predict Monthly Rent', use_container_width=True, disabled=not predict_enabled)


# --- MAIN AREA (OUTPUT) ---

if submit_button:
    # 1. Look up the coordinates for the selected location
    try:
        coord_row = df_city_filtered[df_city_filtered['location'] == selected_location].iloc[0]
        
        latitude = coord_row['latitude']
        longitude = coord_row['longitude']
    except IndexError:
        st.error("Could not find coordinates for the selected location. Data error.")
        st.stop()
    
    # Get the active model for the selected city
    model = models[selected_city]
    
    # Create the input DataFrame
    input_data = pd.DataFrame({
        'house_size': [house_size], 
        'numBathrooms': [num_bathrooms], 
        'numBalconies': [num_balconies],
        'SecurityDeposit': [security_deposit], 
        'bhk': [bhk],
        'latitude': [latitude], 
        'longitude': [longitude],
        'location': [selected_location],
        'Status': [selected_status]
    })
    
    # Display a loading message
    with st.spinner(f'Calculating estimated rent using {selected_city} spatial model...'):
        time.sleep(1) 
        
        try:
            predicted_rent = make_prediction(model, input_data)
            
            st.subheader(f"Estimated Monthly Rent for {selected_location}, {selected_city}")
            st.metric(
                label="Predicted Monthly Rent", 
                value=f"₹ {predicted_rent:,.0f}",
                delta="Spatially Refined Estimate"
            )
            
            st.success(f"Prediction successful using the dedicated {selected_city} model.")
            st.caption(f"Coordinates used: Lat {latitude:.4f}, Long {longitude:.4f}")

        except Exception as e:
            st.error(f"A prediction error occurred. Error: {e}")

else:
    if selected_city is None:
        st.info("First: Select a City to enable all property options.")
    elif selected_location is None:
        st.info("Next: Choose Location to unlock the prediction button.")
    else:
        st.info("Ready? Configure your property details to find its true rental value.")

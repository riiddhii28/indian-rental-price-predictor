import streamlit as st
import pandas as pd
import pickle

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Indian Rental Price Predictor",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .css-1d391kg, .css-1lcbmhc, .css-12oz5g7, .css-1y4p8pa {
        width: 450px !important;
        min-width: 450px !important;
    }
    
    section[data-testid="stSidebar"] {
        width: 450px !important;
        min-width: 450px !important;
    }
    
    section[data-testid="stSidebar"] > div {
        width: 450px !important;
        min-width: 450px !important;
    }
    
    .stAlert > div[data-baseweb="notification"] {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    
    .stError > div[data-baseweb="notification"] {
        background-color: #e3f2fd;
        border-left: 4px solid #1976d2;
    }
    
    .stWarning > div[data-baseweb="notification"] {
        background-color: #e8f4fd;
        border-left: 4px solid #42a5f5;
    }
    
    .stInfo > div[data-baseweb="notification"] {
        background-color: #f3f9ff;
        border-left: 4px solid #64b5f6;
    }
    
    .stButton > button {
        background-color: #2196f3;
        color: white;
        border: none;
        border-radius: 6px;
    }
    
    .stButton > button:hover {
        background-color: #1976d2;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD ASSETS ---
try:
    with open('rf_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    df = pd.read_csv('df_cleaned_for_app.csv')
except FileNotFoundError:
    st.error("Error loading necessary files. Please run `process_data.py` first to generate them.")
    st.stop()

# --- UI LAYOUT ---
st.title("Rental Price Predictor")
st.markdown("Predict the monthly rent for properties in **Delhi, Mumbai, and Pune**.")
st.markdown("---")

# --- SIDEBAR FILTERS ---
st.sidebar.header("Select Your Filters")

# Location Filters - OUTSIDE the form for dynamic updates
st.sidebar.markdown("### Location")
cities = sorted(df['city'].unique())
selected_city = st.sidebar.selectbox('City', cities, index=None, placeholder="Select a city...", key='city_selector')

# Handle location dropdown based on city selection
if selected_city is None:
    # Show disabled location dropdown when no city is selected
    st.sidebar.selectbox('Location', ['Please select a city first'], key='location_selector', disabled=True)
    st.sidebar.warning("Please select a city to see available locations")
    selected_location = None
else:
    # Filter locations based on selected city
    locations_in_city = sorted(df[df['city'] == selected_city]['location'].unique())
    selected_location = st.sidebar.selectbox('Location', locations_in_city, key='location_selector')
    
    # Display helpful info about the selected city
    st.sidebar.info(f"{len(locations_in_city)} locations available in {selected_city}")

# Using a form for other inputs and the button
with st.sidebar.form(key='prediction_form'):

    st.sidebar.markdown("---")

    house_size = st.slider('House Size (sq ft)', 200, 8000, 1200, 50)
    bhk = st.slider('Number of Bedrooms (BHK)', 1, 10, 2)
    num_bathrooms = st.slider('Number of Bathrooms', 1, 10, 2)
    statuses = sorted(df['Status'].unique())
    selected_status = st.selectbox('Furnishing Status', statuses)

    # Submit button for the form
    submit_button = st.form_submit_button(label='Predict Rent', use_container_width=True)


# --- PREDICTION LOGIC ---
if submit_button:
    # Check if city and location are properly selected
    if selected_city is None or selected_location is None:
        st.error("Please select both a city and location before making a prediction.")
    else:
        try:
            # Encode user inputs
            city_encoded = encoders['city'].transform([selected_city])[0]
            location_encoded = encoders['location'].transform([selected_location])[0]
            status_encoded = encoders['Status'].transform([selected_status])[0]

            # Create input DataFrame for the model
            input_data = pd.DataFrame({
                'city': [city_encoded], 'location': [location_encoded],
                'house_size': [house_size], 'bhk': [bhk],
                'numBathrooms': [num_bathrooms], 'Status': [status_encoded]
            })

            # Make prediction
            prediction = model.predict(input_data)[0]

            # Display result in the main area
            st.subheader("Predicted Monthly Rent")
            st.metric(label="Estimated Price", value=f"₹ {prediction:,.0f}")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

else:
    if selected_city is None:
        st.info("Please select a city and location in the sidebar to start.")
    else:
        st.info("Please complete your selections in the sidebar and click 'Predict Rent'.")

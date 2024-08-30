import pandas as pd
import joblib
from xgboost import XGBRegressor
import streamlit as st

# Loading the saved model and preprocessing information
xgb_model = joblib.load('xgboost_model.pkl')
encoded_columns = joblib.load('encoded_columns.pkl')

# Loading the dataset
df_cars_initial = pd.read_csv("Cleaned_data.csv")

df_cars = df_cars_initial.copy()

# Define categorical columns and extract unique values
categorical_columns = ['ft', 'transmission', 'oem', 'model', 'Insurance Validity', 'Color',
                       'Location', 'RTO_grouped']

unique_values = {col: df_cars[col].unique().tolist() for col in categorical_columns}

# Create a mapping of brands to their models
brand_model_mapping = df_cars.groupby('oem')['model'].unique().to_dict()


def preprocess_input(data, encoded_columns, categorical_columns):
    # Apply one-hot encoding
    data_encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

    # Reindex the columns to ensure the same columns as training data
    data_encoded = data_encoded.reindex(columns=encoded_columns, fill_value=0)

    return data_encoded


def predict_price(input_data):
    # Preprocess the input data
    processed_input = preprocess_input(input_data, encoded_columns, categorical_columns)

    # Make prediction
    prediction = xgb_model.predict(processed_input)

    return prediction[0]


def format_inr(number):
    s, *d = str(number).partition(".")
    r = ",".join([s[x-2:x] for x in range(-3, -len(s), -2)][::-1] + [s[-3:]])
    return "".join([r] + d)


def main():
    st.set_page_config(page_title="Car Dheko: Used Car Price Predictor", page_icon="🚗")

    st.markdown('''
        <style>
            body {
                background-color: #E6E6FA;  /* Light purple color */
                text-align: center; /* Center text */
            }
            .sidebar .sidebar-content {
                background-color: #D8BFD8;  /* Thistle color */
                color: white;
                text-align: center;  /* Center text in sidebar */
            }
            .stSelectbox label, .stNumberInput label, .stCheckbox label, .stSlider label {
                font-weight: bold;
                color: #4B0082;  /* Indigo color */
            }
            .stButton>button {
                background-color: #8A2BE2;  /* Violet color */
                color: white;
                border-radius: 10px;
                width: 100%;  /* Make the button full-width */
            }
            .stSidebar {
                display: flex;
                justify-content: center;
                align-items: center;
            }
            .stSidebar .sidebar-content {
                display: flex;
                flex-direction: column;
                align-items: center;
            }
            .result-container {
                background-color: #f0f8ff;  /* Alice blue background for results */
                border: 2px solid #8A2BE2;  /* Violet border */
                border-radius: 15px;
                padding: 20px;
                margin-top: 20px;
                text-align: center;
            }
            .result-container h2 {
                color: #4CAF50;  /* Green color for title */
                font-weight: bold;
            }
            .result-container h1 {
                color: #FF4500;  /* Orange-red color for price */
                font-weight: bold;
            }
            .price-range {
                background-color: #e6e6fa;  /* Light purple color for price range */
                border: 1px solid #4B0082;  /* Indigo border */
                border-radius: 10px;
                padding: 10px;
                margin: 10px;
                color: #4B0082;  /* Indigo text color */
                font-weight: bold;
            }
        </style>
    ''', unsafe_allow_html=True)

    st.title('🚗 Car Dheko: Used Car Price Predictor')

    # Create input fields for the user
    st.sidebar.header('Enter Car Details')

    # Input for Brand and dynamically populate Models
    selected_brand = st.sidebar.selectbox('Brand', unique_values['oem'])
    if selected_brand in brand_model_mapping:
        models = brand_model_mapping[selected_brand].tolist()

    selected_model = st.sidebar.selectbox('Model', models)

    # Input for other categorical and numerical features
    color = st.sidebar.selectbox('Color', unique_values['Color'])
    transmission = st.sidebar.selectbox('Transmission', unique_values['transmission'])
    ft = st.sidebar.selectbox('Fuel Type', unique_values['ft'])
    insurance_validity = st.sidebar.selectbox('Insurance Validity', unique_values['Insurance Validity'])
    location = st.sidebar.selectbox('Location', unique_values['Location'])
    rto_grouped = st.sidebar.selectbox('RTO Grouped', unique_values['RTO_grouped'])

    modelYear = st.sidebar.number_input('Model Year', min_value=2000, max_value=2024)
    turbo_charger = st.sidebar.checkbox('Turbo Charger')
    km = st.sidebar.slider('Kms driven', min_value=0, max_value=300000, step=1000, value=50000)
    engineCC = st.sidebar.slider('Engine CC', min_value=500, max_value=5000, step=100, value=1500)

    # Prepare the input data
    input_data = pd.DataFrame({
        'oem': [selected_brand],
        'model': [selected_model],
        'Color': [color],
        'ft': [ft],
        'transmission': [transmission],
        'Insurance Validity': [insurance_validity],
        'Location': [location],
        'RTO_grouped': [rto_grouped],
        'modelYear': [modelYear],
        'km': [km],
        'Displacement': [engineCC],
        'Turbo Charger': [turbo_charger]
    })

    if st.sidebar.button('Predict'):
        prediction = predict_price(input_data)
        formatted_price = format_inr(prediction)

        st.markdown(f'''
            <div class="result-container">
                <h2>The predicted price for the car is:</h2>
                <h1>₹ {formatted_price}</h1>
            </div>
        ''', unsafe_allow_html=True)

        # Display min and max price range for the selected model
        matching_cars = df_cars[df_cars['model'] == selected_model]
        if not matching_cars.empty:
            min_price = matching_cars['price'].min()
            max_price = matching_cars['price'].max()

            st.markdown(f'''
                <div class="price-range">
                    <strong>Price Range for {selected_model} in Inventory</strong><br>
                    Min Price: ₹ {format_inr(min_price)}<br>
                    Max Price: ₹ {format_inr(max_price)}
                </div>
            ''', unsafe_allow_html=True)
        else:
            st.write(f'No available cars found for the model: {selected_model}')


if __name__ == '__main__':
    main()

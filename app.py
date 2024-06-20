import streamlit as st
import pickle
import numpy as np
from RockvsMine import sonar_data

# Load the saved model
with open('rock vs mine.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Function to make predictions
def predict(input_data):
    input_data_as_nparray = np.asarray(input_data)
    data_reshaped = input_data_as_nparray.reshape(1, -1)
    prediction = loaded_model.predict(data_reshaped)
    return prediction[0]



# Title of the app
st.title('Rock or Mine Prediction App 🔮')


input_data = (
    0.0249, 0.0119, 0.0277, 0.0760, 0.1218, 0.1538, 0.1192, 0.1229, 0.2119,
    0.2531, 0.2855, 0.2961, 0.3341, 0.4287, 0.5205, 0.6087, 0.7236, 0.7577,
    0.7726, 0.8098, 0.8995, 0.9247, 0.9365, 0.9853, 0.9776, 1.0000, 0.9896,
    0.9076, 0.7306, 0.5758, 0.4469, 0.3719, 0.2079, 0.0955, 0.0488, 0.1406,
    0.2554, 0.2054, 0.1614, 0.2232, 0.1773, 0.2293, 0.2521, 0.1464, 0.0673,
    0.0965, 0.1492, 0.1128, 0.0463, 0.0193, 0.0140, 0.0027, 0.0068, 0.0150,
    0.0012, 0.0133, 0.0048, 0.0244, 0.0077, 0.0074
)


if st.button('Predict'):
    result = predict(input_data)
    if result == 'R':
        st.write("This is a Rock 🪨")
    if result == 'M':
        st.write("It is a Mine 🐟")


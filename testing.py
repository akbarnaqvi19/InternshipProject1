## Ali Akbar Naqvi
## Internship Project 1
## California House Price Prediction

import pickle
import numpy as np

with open('CHPPM.pkl', 'rb') as file:
    model=pickle.load(file)
def testing_through_input():
    print("Ocean Proximity to Be Entered in Binary Format")
    test_input = input("Enter Features in Order Longitude, Latitude, Housing Median Age, Total Rooms, Total Bedrooms, Population, Households, Median Income, <1H Ocean, Inland, Island, Near Bay, Naer Ocean : ")

    test_input_list = [float(a) for a in test_input.split(',')]

    input_array = np.array(test_input_list).reshape(1, -1)

    prediction = model.predict(input_array)

    print(f"Predicted Value of House : {prediction[0]}")

testing_through_input()
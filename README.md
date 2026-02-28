# House Price Prediction Web App

This project is a web application that predicts house prices using machine learning, built with Django.

## Features
- User-friendly interface to input house details.
- Real-time house price prediction powered by a trained machine learning model.
- Clean and responsive UI for ease of use.
- Stores prediction history in the database
- Displays previous predictions for review

## Technologies Used
- Django (backend & frontend)
- Python libraries: pandas, numpy, scikit-learn
- HTML/CSS/Bootstrap (frontend styling)

## How It Works
1. The user inputs property details on the web form.
2. The data is sent to the Django backend.
3. The backend processes the input and feeds it to the trained ML model.
4. The model predicts the house price.
5. The predicted price is displayed to the user.

## Installation & Setup
1. Clone the repository
2. Create a virtual environment and activate it.
3. Make sure you have to use your own django key in Settings.py.
4. Write Command : python manage.py runserver 

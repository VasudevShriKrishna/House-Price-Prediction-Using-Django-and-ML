# currently working
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib

data = pd.read_csv("./india_housing_prices.csv")
features = ['City', 'BHK', 'Size_in_SqFt', 'Age_of_Property', 'Furnished_Status']
target = 'Price_in_Lakhs'

df = data[features + [target]].dropna()

# 3. Encode Categorical Variables
city_mapping = {city: idx for idx, city in enumerate(df['City'].unique())}
furnishing_mapping = {"Unfurnished": 0, "Semi-furnished": 1, "Furnished": 2, "Fully furnished": 2} # Added mapping variations

df['City'] = df['City'].map(city_mapping)
df['Furnished_Status'] = df['Furnished_Status'].map(furnishing_mapping)

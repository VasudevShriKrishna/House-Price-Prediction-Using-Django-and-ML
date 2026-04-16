# currently working dont use it before completion
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib

data = pd.read_csv("./india_housing_prices.csv")
features = ['City', 'BHK', 'Size_in_SqFt', 'Age_of_Property', 'Furnished_Status']
target = 'Price_in_Lakhs'

df = data[features + [target]].dropna()
city_mapping = {city: idx for idx, city in enumerate(df['City'].unique())}
furnishing_mapping = {"Unfurnished": 0, "Semi-furnished": 1, "Furnished": 2, "Fully furnished": 2} # Added mapping variations

df['City'] = df['City'].map(city_mapping)
df['Furnished_Status'] = df['Furnished_Status'].map(furnishing_mapping)

X = df.drop(target, axis=1)
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = r2_score(y_test, predictions)
print(f"Model trained with R-squared Accuracy: {accuracy * 100:.2f}%")

joblib.dump(model, "house_model.pkl")
joblib.dump(city_mapping, "city_mapping.pkl")
print("Model and mappings saved successfully!")

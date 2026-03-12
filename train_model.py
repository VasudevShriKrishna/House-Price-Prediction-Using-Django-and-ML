# currently working
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib

data = pd.read_csv("./india_housing_prices.csv")

# 2. Select matching features based on your CSV image
# We will use: City, BHK, Size_in_SqFt, Age_of_Property, Furnished_Status
features = ['City', 'BHK', 'Size_in_SqFt', 'Age_of_Property', 'Furnished_Status']
target = 'Price_in_Lakhs'

df = data[features + [target]].dropna()

# 3. Encode Categorical Variables
city_mapping = {city: idx for idx, city in enumerate(df['City'].unique())}
furnishing_mapping = {"Unfurnished": 0, "Semi-furnished": 1, "Furnished": 2, "Fully furnished": 2} # Added mapping variations

df['City'] = df['City'].map(city_mapping)
df['Furnished_Status'] = df['Furnished_Status'].map(furnishing_mapping)

# 4. Split Data
X = df.drop(target, axis=1)
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train an Advanced Model (Random Forest for 95%+ Accuracy potential)
model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
model.fit(X_train, y_train)

# 6. Check Accuracy
predictions = model.predict(X_test)
accuracy = r2_score(y_test, predictions)
print(f"Model trained with R-squared Accuracy: {accuracy * 100:.2f}%")

# 7. Save the model and mappings
joblib.dump(model, "house_model.pkl")
joblib.dump(city_mapping, "city_mapping.pkl")
print("Model and mappings saved successfully!")

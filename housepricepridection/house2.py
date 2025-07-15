import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1️⃣ Create a dummy dataset
data = data = {
    'City': ['Mumbai', 'Delhi', 'Bengaluru', 'Chennai', 'Ahmedabad', 'Pune', 'Kolkata', 'Hyderabad', 'Himatnagar'],
    'Area_sqft': [1000, 1500, 1200, 1800, 900, 1100, 1600, 1400, 1000],
    'BHK':        [2, 3, 2, 3, 1, 2, 3, 2, 2],
    'Bath':       [1, 2, 2, 3, 1, 2, 2, 2, 2],
    'Price_Lakhs':[100, 50, 120, 70, 80, 110, 140,85, 90]
}


df = pd.DataFrame(data)

# 2️⃣ Define X and y
X = df[['City', 'Area_sqft', 'BHK', 'Bath']]
y = df['Price_Lakhs']

# 3️⃣ Preprocessing - OneHotEncode City column
preprocessor = ColumnTransformer([
    ('onehot', OneHotEncoder(handle_unknown='ignore'), ['City'])
], remainder='passthrough')

# 4️⃣ Build pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# 5️⃣ Train the model
model.fit(X, y)

# 6️⃣ Take user input
print("\n📍 Enter property details to predict the price:")

city = input("City (e.g., Mumbai, Delhi, etc.): ")
area = float(input("Area in square feet: "))
bhk = int(input("Number of bedrooms (BHK): "))
bath = int(input("Number of bathrooms: "))

# 7️⃣ Create input DataFrame
user_input = pd.DataFrame([[city, area, bhk, bath]], columns=['City', 'Area_sqft', 'BHK', 'Bath'])

# 8️⃣ Predict price
predicted_price = model.predict(user_input)[0]
print(f"\n💰 Estimated House Price in {city}: ₹{predicted_price:.2f} Lakhs")

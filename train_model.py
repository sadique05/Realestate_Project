
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor

# Load data
df = pd.read_excel('goregaon_data.xlsx')

# Preprocess
df['bhk'] = df['bhk'].str.extract('(\\d+)').astype(int)

from sklearn.preprocessing import LabelEncoder
le_property = LabelEncoder()
le_furnished = LabelEncoder()
le_location = LabelEncoder()

df['property_type_enc'] = le_property.fit_transform(df['property_type'])
df['furnished_enc'] = le_furnished.fit_transform(df['furnished'])
df['location_enc'] = le_location.fit_transform(df['location'])

X = df[['area', 'bhk', 'price_per_sqft', 'furnished_enc', 'property_type_enc', 'location_enc']]
y = df['price']

# Train model
model = RandomForestRegressor()
model.fit(X, y)

# Save model
import joblib
joblib.dump(model, 'house_price_model.pkl')
joblib.dump(le_furnished, 'le_furnished.pkl')
joblib.dump(le_property, 'le_property.pkl')
joblib.dump(le_location, 'le_location.pkl')

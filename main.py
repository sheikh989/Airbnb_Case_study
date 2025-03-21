from fastapi import FastAPI, Request
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np


app = FastAPI()

file = open("airbnb_model.pkl", 'rb')
model = pickle.load(file)

file_s= open("scaler_airbnb_right.pkl", 'rb')
scl = pickle.load(file_s)

city= open("city_labeling.pkl", 'rb')
city_labeling = pickle.load(city)

zip= open("zip_labeling.pkl", 'rb')
zip_labeling = pickle.load(zip)

pt= open("pt_labeling.pkl", 'rb')
pt_labeling = pickle.load(pt)

rt= open("rt_labeling.pkl", 'rb')
rt_labeling = pickle.load(rt)

lg= open("lg_labeling.pkl", 'rb')
lg_labeling = pickle.load(lg)

bt= open("bt_labeling.pkl", 'rb')
bt_labeling = pickle.load(bt)

@app.get('/')
def home():
    return {'message':'This model predicts the price of an Airbnb listing'}

    

@app.post('/predict_data')
async def predict_price(request:Request):
    if request.method=='POST':
        data = await request.json()

    city = data['city']
    zipcode = data['zipcode']
    property_type = data['property_type']
    room_type = data['room_type']
    bed_type = data['bed_type']
    location_group = data['location_group']
    latitude = int(data['latitude'])
    longitude = int(data['longitude'])
    accommodates = int(data['accommodates'])
    bathrooms = int(data['bathrooms'])
    bedrooms = int(data['bedrooms'])
    beds = int(data['beds'])
    square_feet = int(data['square_feet'])
    availability_365 = int(data['availability_365'])
    number_of_reviews = int(data['number_of_reviews'])
    review_scores_rating = int(data['review_scores_rating'])
    review_scores_cleanliness = int(data['review_scores_cleanliness'])
    review_scores_location = int(data['review_scores_location'])
    review_scores_value = int(data['review_scores_value'])
    price_per_person = int(data['price_per_person'])
        
        
        
    data_dict = {
    "city": city,
    "zipcode": zipcode,
    "property_type": property_type,
    "room_type": room_type,
    "bed_type": bed_type,
    "location_group": location_group,
    "latitude": latitude,
    "longitude": longitude,
    "accommodates": accommodates,
    "bathrooms": bathrooms,
    "bedrooms": bedrooms,
    "beds": beds,
    "square_feet": square_feet,
    "availability_365": availability_365,
    "number_of_reviews": number_of_reviews,
    "review_scores_rating": review_scores_rating,
    "review_scores_cleanliness": review_scores_cleanliness,
    "review_scores_location": review_scores_location,
    "review_scores_value": review_scores_value,
    "price_per_person": price_per_person
}
   

    
    x=pd.DataFrame([data_dict])
    # i= np.array([x])
    x["city"]=city_labeling.transform(x["city"])
    x["zipcode"]=zip_labeling.transform(x["zipcode"])
    x["property_type"]=pt_labeling.transform(x["property_type"])
    x["room_type"]=rt_labeling.transform(x["room_type"])
    x["location_group"]=lg_labeling.transform(x["location_group"])
    x["bed_type"]=bt_labeling.transform(x["bed_type"])

    
    x=scl.transform(x)
    z=model.predict(x)
    return {'price':float(z[0])}



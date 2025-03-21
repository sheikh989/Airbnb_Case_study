import streamlit as st
import pickle
import pandas as pd


st.markdown("<h1 style=color:blue;>Welcome to the Airbnb Price Prediction</h1>", unsafe_allow_html=True)

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


city = st.selectbox("Select City",
                    ('New York', 'Brooklyn', 'Queens', 'Long Island City', 'Bronx', 'Bronx, Riverdale ', 'Jackson Heights', 'Astoria', 'Staten Island', 'JACKSON HEIGHTS', 'Wadsworth Terrace, Manhattan, New York', 'Riverdale', 'Rego Park', 'Springfield Gardens', 'Staten Island ', 'BROOKLYN', 'Flushing', 'EAST ELMHURST', 'bayside', 'Ridgewood', 'New York, Brooklyn', 'Brooklyn ', 'Elmhurst', 'Jamaica', 'New York City', 'LIC', 'ridgewood', 'astoria', 'Middle Village', 'Astoria ', 'Bushwick', 'Corona ', 'Woodside', 'Manhattan', 'Richmond Hill', 'brooklyn', 'Forest Hills', 'New York ', 'Woodhaven', 'Ô\x8aÇÒ_´', 'brooklyn ', 'new york ', 'Jackson Heights, Queens', 'NYC', 'Far Rockaway ', 'bklyn', 'Harlem', 'rigewood', 'Belle Harbor', 'bronx', 'new york', 'ROSEDALE ', 'ASTORIA', 'Midtown West New York', 'West Village', 'Ridgewood ', 'Greenpoint, Brooklyn', 'Brooklyn, New York', 'Financial District New York', 'Williamsburg, Brooklyn', 'Clinton Hill, Brooklyn', 'New Yor', 'Ozone Park', 'Fort Green, Brooklyn', 'Queens ', 'Far Rockaway', 'Bronx, NY', 'Brookly,', 'Queens Village', ' Bushwick,Brooklyn', 'queens', 'BRONX ', 'NYC, Woodhaven Queens', 'Nueva York', 'Prospect Lefferts Garden', 'NY', 'Rockaway Beach', 'jackson heights', 'Rockaway', 'Brighton Beach', 'East Elmhurst', 'College Point', 'Kew Gardens', 'Astoria, Queens ', ' Astoria ', 'Corona', 'Astoria, Queens', 'Yonkers', 'Kingsbridge, NY', 'Sunnyside', ' Astoria', 'Astoria, Queens, NY, USA', 'LONG ISLAND CITY', 'Williamsburg', 'NEW YORK', 'Long Island City ', 'Park Slope, Brooklyn', 'Union Square, East Village, New York', 'ASTORIA ', 'Forest Hills, Queens', 'Times Square', 'Astoria (Queens)', 'Bushwick, Brooklyn', 'Saint Albans(Queens)', 'Brooklyn NYC', 'Suunyside', 'Roosevelt Island', 'Howard Beach', 'Carroll Gardens,Brooklyn, NYC', 'Kew Gardens, Queens', 'Maspeth', 'Fort Greene, Brooklyn', 'Whitestone', 'rockaway park  boro of queens', 'East Williamsburg', 'Rockaway Park', 'Hamilton Heights, New York', 'jamaica estates', 'long island city', 'Bayside', 'Cypress Hills, Brooklyn', 'Ozon Park', 'sunnnyside', 'New York/Astoria', 'Lic', 'Windsor Terrace', ' Woodside ', 'rockaway beach ', 'Rockaway beach', 'Fresh meadows', 'BRONX', 'Bronx ', 'Upper East Side, New York', 'Astoria,New York ', 'Long Island city', 'Williamsburg Brooklyn', 'Woodside Queens', 'Astoria, New York ', 'Astoria, NYC', 'New york ', 'Elmhurst ', 'Bx', 'Arverne', 'woodside', 'Nyc', 'Brooklyn, Bedford Stuyvesant', 'Greenpoint', 'Briarwood', 'Brooklyn Kensington area', 'SoHo New York', 'Brooklyn\ran/ll/lol/', 'Lower East Side', 'JACKSON HEIGHT', 'maspeth', 'Brooklyn\rBrooklyn sheepshead bay', 'bronx ', 'brooklyn, NY', 'ridgewood queens', 'Chelsea, New York', 'Brooklyn,  Ny 11221', 'woodhaven boulevard ', 'Kips Bay', 'lonf island city', 'Sobro', 'Chelsea', 'Jackson heights ', 'Downtown Brooklyn ', 'Briarwood  Queens, NYC', 'Fort Greene', 'Auburndale', 'Astoria queens', '_ïãóã\x80___é_\x9f__', 'Brooklyn. ', '_\x9dãëã_-_»__ãó__', 'Ny ', 'New york', 'Ny', 'Bklyn. ', 'NYC ', 'Washington heights', 'Manhattan, NY', 'South Richmond Hill', 'Long Island city ', 'NY ', '\x8d__\x8d__'),
                    index=0,placeholder='Select City')
zipcode = st.selectbox("Select Zipcode",
                    ('10022-4175', '11211', '11221', '10011', '11231', '11207', '10013', '10003', '11217', '10018', '11213', '10019', '10014', '10040', '10033', '11238', '10038', '10027', '11222', '11206', '10025', '10030', '10035', '10009', '10031', '10016', '10026', '10005', '10012', '11102', '10128', '11101', '11385', '10028', '11215', '10007', '11205', '10023', '10002', '10010', '11237', '10024', '10039', '11233', '10022', '10037', '10455', '11220', '11249', '11216', '10075', '11235', '11201', '11374', '10036', '11234', '10001', '10032', '11103', '10463', '11106', '11230', '10466', '11372', '10017', '11377', '11109', '11105', '10301', '11210', '10065', '11226', '10021', '11373', '10004', '11432', '11415', '10306', '10034', '11413', '11236', '10029', '11225', '11365', '10305', '11355', '11375', '11204', '11369', '11208', '11370', '11361', '10464', '11378', '11223', '11104', '11203', '10451', '11358', '11421', '11433', '10467', '11232', '10006', '11218', '10280', '11209', '11368', '11435', '10461', '11379', '10282', '10459', '10458', '11694', '11224', '11420', '10044', '10473', '10469', '11229', '10456', '11418', '11367', '10162', '11411', '11693', '10303', '11436', '11691', '10457', '10453', '10452', '10454', '11434', '11692', '10304', '11417', '10069', '11429', '10080', '11422', '11354', '10460', '11427', '10468', '11214', '10471', '10314', '11219', '10472', '11126', '11239', '11212', '11419', '11356', '10475', '11416', '11412', '10462', '11366', '10470', '111006', '10704', '10465', '11228', '10003-6521', '11360', '10281', '1001', '11414', '11357', '11423', '10302', '10308', '1003', '10312', '11364', '10307', '10310', '10003-8623', '14072', '8456422473 call for more details', '10045', '12512'),
                    index=0,placeholder='Select Zipcode')
property_type = st.selectbox("Select Property Type",('Apartment', 'House', 'Loft', 'Dorm', 'Bed & Breakfast', 'Other', 'Hut', 'Treehouse', 'Boat', 'Tent', 'Castle', 'Cabin', 'Chalet', 'Villa', 'Cave', 'Earth House', 'Lighthouse', 'Camper/RV'),
                    index=0,placeholder='Select Property Type')         
room_type = st.selectbox("Select Room Type",('Entire home/apt', 'Private room', 'Shared room'),
                    index=0,placeholder='Select Room Type')     
bed_type = st.selectbox("Select Bed Type",('Real Bed', 'Futon', 'Pull-out Sofa', 'Airbed', 'Couch'),
                    index=0,placeholder='Select Bed Type')
location_group = st.selectbox("select Location Group",('3_2', '4_2', '2_3', '2_2', '1_2', '4_3', '3_3', '0_2', '1_3', '1_1', '2_4', '0_1', '1_4', '3_4', '4_4', '0_3', '0_4', '1_0', '0_0')
                              ,index=0,placeholder='Select Location Group') 

latitude = st.number_input("Enter Latitude",value=0.0)
longitude = st.number_input("Enter Longitude",value=0.0)
accommodates = st.slider("Select Accommodates",1,16)
bathrooms = st.slider("Select Bathrooms",1,8)
bedrooms = st.slider("Select Bedrooms",1,10)
beds = st.slider("Select Beds",1,16) 
square_feet = st.number_input("Enter Square Feet",value=0)
availability_365 = st.slider("Select Availability",1,365)
number_of_reviews = st.slider("Select Number of Reviews",1,220) 
review_scores_rating = st.slider("Select Review Scores Rating",20,100)
review_scores_cleanliness = st.slider("Select Review Scores Cleanliness",1,10)
review_scores_location = st.slider("Select Review Scores Location",1,10)
review_scores_value = st.slider("Select Review Scores Value",1,10)
price_per_person = st.number_input("Enter Price Per Person",value=0,)

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
x["city"]=city_labeling.transform(x["city"])
x["zipcode"]=zip_labeling.transform(x["zipcode"])
x["property_type"]=pt_labeling.transform(x["property_type"])
x["room_type"]=rt_labeling.transform(x["room_type"])
x["location_group"]=lg_labeling.transform(x["location_group"])
x["bed_type"]=bt_labeling.transform(x["bed_type"])

    
x=scl.transform(x)
z=model.predict(x)

if st.button("Predict Price"):
    st.success(f"Price is {float(z)}")


# st.title("hiii you can doo")
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor
from boruta import BorutaPy


########################################################

'''
WEB APPLICATION
'''
import streamlit as st

st.write('Estimation of Permeability Reduction due to Asphaltene Deposition By Extra Trees Algorithm')

st.sidebar.header('User input parameters')

def user_features():
    API = st.sidebar.slider('API', 21.0, 52.89, 30.0)
    core_area = st.sidebar.slider('Core Area (cm^2)', 4.15, 11.41, 7.00)
    inj_velocity = st.sidebar.slider('Injection Velocity (cm/hr)', 0.0, 35.0, 15.0)
    asph_content = st.sidebar.slider('Asphaltene Content (wt. %)', 0.0, 17.0, 10.0)
    RockType = st.sidebar.selectbox('Rock Type',['Carbonate', 'SandStone'])
    Porosity = st.sidebar.slider('Porosity (%)', 7.0, 35.0, 20.0)
    permeability = st.sidebar.slider('Permeability (mD)', 0.0, 107.0, 55.0)
    core_length = st.sidebar.slider('Core Length (cm)', 4.5, 26.5, 10.0)
    temp = st.sidebar.slider('Temperature (c)', 22.0, 99.0, 50.0)
    pore_inj = st.sidebar.slider('Pore Volume Injected', 0.0, 746.0, 100.0)
    
    data_web = {
        'API': API, 'Core Area': core_area, 'Injection Velocity': inj_velocity, 'Asphaltene Content': asph_content,
        'Rock Type': RockType, 'Porosity': Porosity, 'Permeabilty': permeability, 'Core Length': core_length,
        'Temperature':temp, 'Pore Volume Injected': pore_inj
                }
    
    features = pd.DataFrame(data_web, index=[0])
    return features

df_web = user_features()

st.subheader('User Defined Features')
st.write(df_web)

#######################

data = pd.read_csv(r"D:\MASTER _ Production Eng\THESIS _ Machine Learning\Main Folder\Codes & DataSets\DataSet _ Natural Depletion.csv")
df = data.drop(['Source', 'Year', 'FirstAuthor', 'Injection FlowRate (cc/hour)'], axis=1)

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

encoder.fit(df.RockType)
df.RockType = encoder.transform(df.RockType) # carbonate = 0 , sandstone = 1

x = df.drop('K/K0', axis=1)
y = df['K/K0'].values
feature_names = np.array(x.columns)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)


xgb = XGBRegressor()
feat_selector = BorutaPy(xgb, n_estimators='auto', verbose=2, random_state=0)
feat_selector.fit(x_train_scaled, y_train)

x_train_filtered = feat_selector.transform(x_train_scaled)
x_test_filtered = feat_selector.transform(x_test_scaled)

model = ExtraTreesRegressor(n_estimators= 135, criterion= 'mse', min_samples_split= 2, 
min_samples_leaf= 1, min_weight_fraction_leaf= 0.00015154255722980094, 
max_depth= 50, max_features= None, random_state=0)

X_scaled = scaler.transform(x)
X_filtered = feat_selector.transform(X_scaled)

model.fit(X_filtered, y)
######################################

df_web['Rock Type'] = encoder.transform(df_web['Rock Type'])

df_web_scaled = scaler.transform(df_web)
df_web_filtered = feat_selector.transform(df_web_scaled)

prediction = model.predict(df_web_filtered)

st.subheader('K/Ki Prediction')
st.write(prediction)

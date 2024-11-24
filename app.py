import streamlit as st
import openmeteo_requests
import requests_cache
import pandas as pd
import accelerate
from retry_requests import retry
from geopy.geocoders import Nominatim
from transformers import pipeline
from huggingface_hub import login
import torch
import os

urls = [
    "https://api.open-meteo.com/v1/forecast",
    "https://air-quality-api.open-meteo.com/v1/air-quality"]
forecast_params  = ['temperature_2m_max', 'temperature_2m_min', 'apparent_temperature_max', 'apparent_temperature_min', 'precipitation_sum', 'rain_sum', 'showers_sum', 'precipitation_hours', 'precipitation_probability_mean', 'sunrise', 'sunset', 'wind_speed_10m_max', 'wind_gusts_10m_max', 'wind_direction_10m_dominant']
air_params = ['carbon_monoxide', 'carbon_dioxide', 'ammonia', 'methane', 'dust', 'us_aqi']
web_params = [forecast_params, air_params]

def func(x):
    return x.upper().replace('_',' ')

def to_dict(idx, data):
    dct = dict()
    if data == None:
        return
    for l, item in enumerate(web_params[idx]):
        dct[item] = data.Variables(l).ValuesAsNumpy()
    return dct

def to_df(data):
    df = pd.DataFrame(data)
    df.columns = df.columns.to_series().apply(func)
    return df

@st.cache_data
def get_weather(place_name):
    geolocator = Nominatim(user_agent="noma")
    location = geolocator.geocode(place_name)
    city_health = dict()
    city_health.update({'Location': place_name})
    cords = {
        'lat': location.latitude,
        'lon': location.longitude
    }
    for idx, url in enumerate(urls):
        params = {
            'latitude': cords['lat'],
            'longitude': cords['lon'],
            'timezone': 'auto',
            'daily': web_params[idx]
        }
        if idx == 0:
            params['temperature_unit'] = 'fahrenheit'
            params['wind_speed_unit'] = 'mph'
            params['precipitation_unit'] = 'inch'

        cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
        retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
        openmeteo = openmeteo_requests.Client(session = retry_session)
        responses = openmeteo.weather_api(url, params=params)
        if responses != None:
            response = responses[0]
            daily = response.Daily()
            data = to_dict(idx, daily)
            df = to_df(data)
            key = url.rsplit('/', 1)[1]
            city_health.update({key: df})

    return city_health

@st.cache_data
def load_llama():
    os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]
    hf_token = os.getenv("HF_TOKEN")
    login(hf_token)
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    pipe = pipeline(
        "text-generation",
        model=model_id,
    )

    return pipe

def get_description(city):
    pipe = load_llama()
    messages = [
        {"role": "system", "content": '''
            You are a chatbot that takes in a place name and a brief history of the place as well as a bit 
            of current events that have happened lately.'''},
        {"role": "user", "content": city},
    ]
    outputs = pipe(
        messages,
        max_new_tokens=256,
    )
    return outputs[0]["generated_text"][-1]

st.set_page_config(initial_sidebar_state="collapsed")
st.title('Better Weather')
place_name = st.text_input('Enter a place name')
if st.button("Get Weather"):
    weather_data = get_weather(place_name)
    place_description = get_description(place_name)
    with st.expander("Weather Data"):
        for key, df in weather_data.items():
            if type(df) == pd.DataFrame:
                if not df.empty:
                    st.subheader(key.title())
                    if type(df) == pd.DataFrame:
                        st.dataframe(df)
            else:
                st.subheader(key.title())
                st.write(place_description)

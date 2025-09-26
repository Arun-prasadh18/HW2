import requests
import streamlit as st
import openai
import json

st.title("Weather & Clothing Advisor")

# Get weather data
def get_current_weather(location):
    if not location:
        location = "Syracuse NY"
    
    if "," in location:
        location = location.split(",")[0].strip()
    
    API_key = st.secrets["Openweather_API_key"]
    url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={API_key}"
    
    response = requests.get(url)
    data = response.json()
    
    temp = data['main']['temp'] - 273.15
    feels_like = data['main']['feels_like'] - 273.15
    humidity = data['main']['humidity']
    description = data['weather'][0]['description']
    
    return {
        "location": location,
        "temperature": round(temp, 2),
        "feels_like": round(feels_like, 2),
        "humidity": humidity,
        "description": description
    }

# Weather function for OpenAI
def weather_function(location):
    weather_data = get_current_weather(location)
    return json.dumps(weather_data)

# Function schema
weather_tool = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"}
            },
            "required": ["location"]
        }
    }
}

# Main function
def get_suggestions(city):
    client = openai.OpenAI(api_key=st.secrets["API_KEY"])
    
    # First call - with weather tool
    response1 = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": f"What should I wear in {city}?"}],
        tools=[weather_tool],
        tool_choice="auto"
    )
    
    # If weather function was called
    if response1.choices[0].message.tool_calls:
        tool_call = response1.choices[0].message.tool_calls[0]
        location = json.loads(tool_call.function.arguments)["location"]
        weather_data = weather_function(location)
        
        # Second call - with weather info
        response2 = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "user", 
                "content": f"Weather data: {weather_data}. Give clothing suggestions for this weather."
            }]
        )
        
        return response2.choices[0].message.content, json.loads(weather_data)
    
    return response1.choices[0].message.content, None

# UI
city = st.text_input("Enter city name:")

if st.button("Get Suggestions"):
    if city:
        suggestions, weather = get_suggestions(city)
        
        if weather:
            st.write(f"**Weather in {weather['location']}:**")
            st.write(f"Temperature: {weather['temperature']}°C")
            st.write(f"Description: {weather['description']}")
            st.write("---")
        
        st.write("**Clothing Suggestions:**")
        st.write(suggestions)
    else:
        st.write("Please enter a city name!")
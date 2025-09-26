# import requests
# import streamlit as st
# def get_current_weather(location, API_key):
#     if "," in location:
#         location = location.split(",")[0].strip()
#         urlbase = "https://api.openweathermap.org/data/2.5/"
#         urlweather = f"weather?q={location}&appid={API_key}"
#         url = urlbase + urlweather
#         response = requests.get(url)
#         data = response.json()
#         # Extract temperatures & Convert Kelvin to Celsius
#         temp = data['main']['temp'] - 273.15
#         feels_like = data['main']['feels_like'] - 273.15
#         temp_min = data['main']['temp_min'] - 273.15
#         temp_max = data['main']['temp_max'] - 273.15
#         humidity = data['main']['humidity']
#         return {"location": location,
#         "temperature": round(temp, 2),
#         "feels_like": round(feels_like, 2),
#         "temp_min": round(temp_min, 2),
#         "temp_max": round(temp_max, 2),
#         "humidity": round(humidity, 2)
#         }
# weathter_api_key = st.secrets["Openweather_API_key"]
# print(get_current_weather('Syracuse, NY', weathter_api_key))
# import requests
# import streamlit as st

# def get_current_weather(location, API_key):
#     # Remove the if statement - process all locations the same way
#     location = location.strip()
    
#     urlbase = "https://api.openweathermap.org/data/2.5/"
#     urlweather = f"weather?q={location}&appid={API_key}"
#     url = urlbase + urlweather
    
#     response = requests.get(url)
#     data = response.json()
    
#     # Check if the request was successful
#     if response.status_code != 200 or 'main' not in data:
#         print(f"Error: {data.get('message', 'Failed to fetch weather data')}")
#         return None
    
#     # Extract temperatures & Convert Kelvin to Celsius
#     temp = data['main']['temp'] - 273.15
#     feels_like = data['main']['feels_like'] - 273.15
#     temp_min = data['main']['temp_min'] - 273.15
#     temp_max = data['main']['temp_max'] - 273.15
#     humidity = data['main']['humidity']
    
#     return {
#         "location": location,
#         "temperature": round(temp, 2),
#         "feels_like": round(feels_like, 2),
#         "temp_min": round(temp_min, 2),
#         "temp_max": round(temp_max, 2),
#         "humidity": round(humidity, 2)
#     }

# # Test the function
# weather_api_key = st.secrets["Openweather_API_key"]
# result = get_current_weather('Syracuse,NY', weather_api_key)

# if result:
#     print(result)
# else:
#     print("Failed to get weather data")
#lab 5b
# import streamlit as st
# import requests
# import json
# from openai import OpenAI

# # --- 1. SETUP: API KEYS & CLIENT ---
# # The script expects your API keys to be stored in Streamlit's secrets management.
# # Create a file at .streamlit/secrets.toml and add your keys:
# #
# # Openweather_API_key = "YOUR_OPENWEATHERMAP_API_KEY"
# # OpenAI_API_key = "YOUR_OPENAI_API_KEY"
# #

# try:
#     openai_api_key = st.secrets["API_KEY"]
#     weather_api_key = st.secrets["Openweather_API_key"]
#     client = OpenAI(api_key=openai_api_key)
# except (FileNotFoundError, KeyError):
#     st.error("API keys not found. Please add them to your Streamlit secrets.")
#     st.stop()

# # --- 2. TOOL DEFINITION: GET CURRENT WEATHER ---
# # This is your function, now defined to be used as a "tool" by the OpenAI model.
# # I've added error handling and a default location as requested.

# def get_current_weather(location: str):
#     """
#     Gets the current weather for a specified location using the OpenWeatherMap API.

#     Args:
#         location (str): The city name. If empty or not provided, defaults to "Syracuse, NY".
    
#     Returns:
#         str: A JSON string containing weather data or an error message.
#     """
#     if not location:
#         location = "Syracuse, NY" # Default location per instructions

#     base_url = "https://api.openweathermap.org/data/2.5/weather"
#     params = {
#         "q": location,
#         "appid": weather_api_key,
#         "units": "metric"  # Request temperature in Celsius directly
#     }

#     try:
#         response = requests.get(base_url, params=params)
#         response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
#         data = response.json()
        
#         # Extract relevant information
#         weather_info = {
#             "location": data["name"],
#             "temperature_celsius": data["main"]["temp"],
#             "feels_like_celsius": data["main"]["feels_like"],
#             "humidity_percent": data["main"]["humidity"],
#             "condition": data["weather"][0]["description"],
#         }
#         return json.dumps(weather_info)

#     except requests.exceptions.HTTPError as http_err:
#         if response.status_code == 404:
#             return json.dumps({"error": f"City '{location}' not found."})
#         return json.dumps({"error": f"An API error occurred: {http_err}"})
#     except Exception as e:
#         return json.dumps({"error": f"An unexpected error occurred: {e}"})

# # --- 3. STREAMLIT USER INTERFACE ---

# st.title("👕 AI Clothing Advisor")
# st.markdown("Enter a city, and I'll suggest what you should wear based on today's weather.")

# # User input for the city
# location_input = st.text_input("Enter a city name:", "Syracuse, NY")

# if st.button("Get Suggestion"):
#     with st.spinner("Checking the weather and picking an outfit..."):
#         final_location = location_input if location_input.strip() else "Syracuse, NY"

#         # Define the tools the model can use
#         tools = [
#             {
#                 "type": "function",
#                 "function": {
#                     "name": "get_current_weather",
#                     "description": "Get the current weather for a given city.",
#                     "parameters": {
#                         "type": "object",
#                         "properties": {
#                             "location": {
#                                 "type": "string",
#                                 "description": "The city name, e.g., 'Paris' or 'Syracuse, NY'",
#                             },
#                         },
#                         "required": ["location"],
#                     },
#                 },
#             }
#         ]
        
#         # The conversation history starts with the user's request
#         # messages = [{"role": "user", "content": f"What clothes should I wear today in {location_input}?"}]
#         messages = [{"role": "user", "content": f"What clothes should I wear today in {final_location}?"}]

#         # --- 4. FIRST API CALL (Decide if a tool is needed) ---
#         # The model sees the user's request and the available tools.
#         # `tool_choice="auto"` lets the model decide whether to call the function.
        
#         try:
#             first_response = client.chat.completions.create(
#                 model="gpt-4o",
#                 messages=messages,
#                 tools=tools,
#                 tool_choice="auto",
#             )
            
#             response_message = first_response.choices[0].message
#             tool_calls = response_message.tool_calls

#             # --- 5. TOOL INVOCATION (If the model chose to use a tool) ---
#             if tool_calls:
#                 # Add the assistant's response (which contains the tool call request) to the history
#                 messages.append(response_message) 

#                 # Execute the function call
#                 for tool_call in tool_calls:
#                     function_name = tool_call.function.name
#                     function_args = json.loads(tool_call.function.arguments)
                    
#                     # Call the actual Python function with the arguments provided by the model
#                     function_response = get_current_weather(
#                         location=function_args.get("location")
#                     )
                    
#                     # Add the function's output to the conversation history
#                     messages.append(
#                         {
#                             "tool_call_id": tool_call.id,
#                             "role": "tool",
#                             "name": function_name,
#                             "content": function_response, # This is the weather data
#                         }
#                     )

#                 # --- 6. SECOND API CALL (Generate final response with weather data) ---
#                 # The model now has the original prompt AND the weather data from the tool.
#                 # It can now generate a human-readable clothing suggestion.
                
#                 second_response = client.chat.completions.create(
#                     model="gpt-4o",
#                     messages=messages,
#                 )
                
#                 final_suggestion = second_response.choices[0].message.content
#                 st.success(final_suggestion)

#             else:
#                 # Fallback in case the model answers directly without using the tool
#                 st.info(response_message.content)

#         except Exception as e:
#             st.error(f"An error occurred with the OpenAI API: {e}")

# import streamlit as st
# import requests
# import json
# from openai import OpenAI

# # --- 1. SETUP: API KEYS & CLIENT ---
# try:
#     openai_api_key = st.secrets["API_KEY"]
#     weather_api_key = st.secrets["Openweather_API_key"]
#     st.write(weather_api_key)
#     client = OpenAI(api_key=openai_api_key)
# except (FileNotFoundError, KeyError):
#     st.error("API keys not found. Please add them to your Streamlit secrets.")
#     st.stop()

# # --- 2. TOOL DEFINITION: GET CURRENT WEATHER ---
# def get_current_weather(location: str):
#     """
#     Gets the current weather for a specified location using the OpenWeatherMap API.
#     """
#     # The default from the previous version is no longer strictly necessary here,
#     # but we'll keep it as a fallback.
#     if not location:
#         location = "Syracuse, NY"

#     base_url = "https://api.openweathermap.org/data/2.5/weather"
#     params = {
#         "q": location,
#         "appid": weather_api_key,
#         "units": "metric"
#     }

#     try:
#         response = requests.get(base_url, params=params)
#         response.raise_for_status()
#         data = response.json()
        
#         weather_info = {
#             "location": data["name"],
#             "temperature_celsius": data["main"]["temp"],
#             "feels_like_celsius": data["main"]["feels_like"],
#             "humidity_percent": data["main"]["humidity"],
#             "condition": data["weather"][0]["description"],
#         }
#         return json.dumps(weather_info)

#     except requests.exceptions.HTTPError as http_err:
#         if response.status_code == 404:
#             return json.dumps({"error": f"City '{location}' not found."})
#         return json.dumps({"error": f"An API error occurred: {http_err}"})
#     except Exception as e:
#         return json.dumps({"error": f"An unexpected error occurred: {e}"})

# # --- 3. STREAMLIT USER INTERFACE ---
# st.title("👕 AI Clothing Advisor")
# st.markdown("Enter a city, and I'll suggest what you should wear based on today's weather.")

# location_input = st.text_input("Enter a city name:", placeholder="e.g., Syracuse, NY")

# if st.button("Get Suggestion"):
#     with st.spinner("Checking the weather and picking an outfit..."):
        
#         # --- THIS IS THE FIX ---
#         # If the user clears the input box, location_input will be an empty string.
#         # We check for this and assign our default location before constructing the API message.
#         final_location = location_input if location_input.strip() else "Syracuse, NY"
        
#         # Define the tools the model can use
#         tools = [
#             {
#                 "type": "function",
#                 "function": {
#                     "name": "get_current_weather",
#                     "description": "Get the current weather for a given city.",
#                     "parameters": {
#                         "type": "object",
#                         "properties": {
#                             "location": {
#                                 "type": "string",
#                                 "description": "The city name, e.g., 'Paris' or 'Syracuse, NY'",
#                             },
#                         },
#                         "required": ["location"],
#                     },
#                 },
#             }
#         ]
        
#         # The conversation history starts with the user's request, now with a guaranteed location
#         messages = [{"role": "user", "content": f"What clothes should I wear today in {final_location}?"}]

#         # --- 4. FIRST API CALL (Decide if a tool is needed) ---
#         try:
#             first_response = client.chat.completions.create(
#                 model="gpt-4o",
#                 messages=messages,
#                 tools=tools,
#                 tool_choice="auto",
#             )
            
#             response_message = first_response.choices[0].message
#             tool_calls = response_message.tool_calls

#             # --- 5. TOOL INVOCATION (If the model chose to use a tool) ---
#             if tool_calls:
#                 messages.append(response_message) 

#                 for tool_call in tool_calls:
#                     function_name = tool_call.function.name
#                     function_args = json.loads(tool_call.function.arguments)
                    
#                     function_response = get_current_weather(
#                         location=function_args.get("location")
#                     )
                    
#                     messages.append(
#                         {
#                             "tool_call_id": tool_call.id,
#                             "role": "tool",
#                             "name": function_name,
#                             "content": function_response,
#                         }
#                     )

#                 # --- 6. SECOND API CALL (Generate final response with weather data) ---
#                 second_response = client.chat.completions.create(
#                     model="gpt-4o",
#                     messages=messages,
#                 )
                
#                 final_suggestion = second_response.choices[0].message.content
#                 st.success(final_suggestion)

#             else:
#                 st.info(response_message.content)

#         except Exception as e:
#             st.error(f"An error occurred with the OpenAI API: {e}")

# import streamlit as st
# import requests
# import json
# from openai import OpenAI

# # Initialize OpenAI client
# client = OpenAI(api_key=st.secrets["API_KEY"])

# def get_current_weather(location):
#     """
#     Fetch current weather data for a given location using OpenWeatherMap API.
    
#     Args:
#         location (str): City name or "City, Country Code" 
    
#     Returns:
#         str: JSON string with weather data
#     """
#     # Use Syracuse, NY as default if no location provided
#     if not location or location.strip() == "":
#         location = "Syracuse, NY"
    
#     API_key = st.secrets["Openweather_API_key"]
    
#     try:
#         # Clean up location string
#         location = location.strip()
        
#         # Build API URL
#         urlbase = "https://api.openweathermap.org/data/2.5/"
#         urlweather = f"weather?q={location}&appid={API_key}"
#         url = urlbase + urlweather
        
#         # Make API request
#         response = requests.get(url)
        
#         # Check if request was successful
#         if response.status_code != 200:
#             return json.dumps({
#                 "error": True,
#                 "message": f"API request failed with status code {response.status_code}",
#                 "location": location
#             })
        
#         data = response.json()
        
#         # Check if the API returned an error
#         if "cod" in data and data["cod"] != 200:
#             return json.dumps({
#                 "error": True,
#                 "message": data.get("message", "Unknown error from API"),
#                 "location": location
#             })
        
#         # Extract temperatures & Convert Kelvin to Celsius
#         temp = data['main']['temp'] - 273.15
#         feels_like = data['main']['feels_like'] - 273.15
#         temp_min = data['main']['temp_min'] - 273.15
#         temp_max = data['main']['temp_max'] - 273.15
#         humidity = data['main']['humidity']
        
#         # Extract additional useful information
#         weather_description = data['weather'][0]['description'] if 'weather' in data else "N/A"
#         weather_main = data['weather'][0]['main'] if 'weather' in data else "N/A"
#         wind_speed = data['wind']['speed'] if 'wind' in data else 0
        
#         # Convert to Fahrenheit as well for US users
#         temp_f = (temp * 9/5) + 32
#         feels_like_f = (feels_like * 9/5) + 32
        
#         weather_info = {
#             "location": location,
#             "country": data['sys'].get('country', 'N/A'),
#             "temperature_celsius": round(temp, 1),
#             "temperature_fahrenheit": round(temp_f, 1),
#             "feels_like_celsius": round(feels_like, 1),
#             "feels_like_fahrenheit": round(feels_like_f, 1),
#             "temp_min_celsius": round(temp_min, 1),
#             "temp_max_celsius": round(temp_max, 1),
#             "humidity": humidity,
#             "weather_condition": weather_main,
#             "description": weather_description,
#             "wind_speed_ms": wind_speed
#         }
        
#         return json.dumps(weather_info)
        
#     except Exception as e:
#         return json.dumps({
#             "error": True,
#             "message": f"Error fetching weather: {str(e)}",
#             "location": location
#         })

# def get_clothing_recommendation(city):
#     """
#     Get weather for a city and provide clothing recommendations using OpenAI.
    
#     Args:
#         city (str): City name
    
#     Returns:
#         str: Clothing recommendation based on weather
#     """
    
#     # Define the function/tool for OpenAI
#     tools = [
#         {
#             "type": "function",
#             "function": {
#                 "name": "get_current_weather",
#                 "description": "Get the current weather in a given location",
#                 "parameters": {
#                     "type": "object",
#                     "properties": {
#                         "location": {
#                             "type": "string",
#                             "description": "The city and state/country, e.g. San Francisco, CA or London, UK"
#                         }
#                     },
#                     "required": ["location"]
#                 }
#             }
#         }
#     ]
    
#     # First API call - Let OpenAI decide if it needs weather data
#     messages = [
#         {
#             "role": "system",
#             "content": "You are a helpful assistant that provides clothing recommendations based on weather conditions."
#         },
#         {
#             "role": "user",
#             "content": f"What clothes should I wear today in {city}?" if city else "What clothes should I wear today?"
#         }
#     ]
    
#     # First call with tool_choice='auto'
#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=messages,
#         tools=tools,
#         tool_choice="auto"
#     )
    
#     first_response = response.choices[0].message
    
#     # Check if the model wants to use the weather function
#     if first_response.tool_calls:
#         # Extract function arguments
#         tool_call = first_response.tool_calls[0]
#         function_args = json.loads(tool_call.function.arguments)
        
#         # Get weather data
#         weather_data = get_current_weather(function_args.get("location", "Syracuse, NY"))
        
#         # Add the assistant's message with tool call to conversation
#         messages.append({
#             "role": "assistant",
#             "content": first_response.content,
#             "tool_calls": [
#                 {
#                     "id": tool_call.id,
#                     "type": "function",
#                     "function": {
#                         "name": "get_current_weather",
#                         "arguments": tool_call.function.arguments
#                     }
#                 }
#             ]
#         })
        
#         # Add the function response to conversation
#         messages.append({
#             "role": "tool",
#             "tool_call_id": tool_call.id,
#             "content": weather_data
#         })
        
#         # Second API call - Get clothing recommendations based on weather
#         weather_dict = json.loads(weather_data)
        
#         if not weather_dict.get("error"):
#             prompt = f"""Based on the following weather conditions, provide specific and practical clothing recommendations:
            
#             Location: {weather_dict.get('location')}
#             Temperature: {weather_dict.get('temperature_fahrenheit')}°F ({weather_dict.get('temperature_celsius')}°C)
#             Feels Like: {weather_dict.get('feels_like_fahrenheit')}°F ({weather_dict.get('feels_like_celsius')}°C)
#             Weather: {weather_dict.get('weather_condition')} - {weather_dict.get('description')}
#             Humidity: {weather_dict.get('humidity')}%
#             Wind Speed: {weather_dict.get('wind_speed_ms')} m/s
            
#             Please provide:
#             1. Specific clothing recommendations (top, bottom, footwear)
#             2. Any accessories needed (umbrella, sunglasses, hat, etc.)
#             3. General comfort tips for these conditions"""
            
#             messages.append({
#                 "role": "user",
#                 "content": prompt
#             })
#         else:
#             messages.append({
#                 "role": "user",
#                 "content": f"There was an error getting weather data: {weather_dict.get('message')}. Please provide general clothing advice for Syracuse, NY based on typical weather for this time of year."
#             })
        
#         # Get final recommendation
#         final_response = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=messages
#         )
        
#         return final_response.choices[0].message.content
#     else:
#         # If no tool was called, return the direct response
#         return first_response.content

# # Streamlit UI
# def main():
#     st.set_page_config(page_title="Weather Clothing Bot", page_icon="🌤️")
    
#     st.title("🌤️ Weather-Based Clothing Recommender")
#     st.markdown("Enter a city to get personalized clothing recommendations based on current weather conditions!")
    
#     # Initialize session state for storing recommendations
#     if 'recommendations' not in st.session_state:
#         st.session_state.recommendations = []
    
#     # Create two columns for better layout
#     col1, col2 = st.columns([2, 1])
    
#     with col1:
#         # Input form
#         with st.form("weather_form"):
#             city_input = st.text_input(
#                 "Enter a city:", 
#                 placeholder="e.g., New York, NY or London, UK",
#                 help="Leave empty to use Syracuse, NY as default"
#             )
            
#             submitted = st.form_submit_button("Get Clothing Recommendations", type="primary")
    
#     with col2:
#         st.info("💡 **Tip:** You can enter cities in formats like:\n- Syracuse, NY\n- London, UK\n- Tokyo\n- Paris, France")
    
#     # Process form submission
#     if submitted:
#         with st.spinner("🔍 Checking weather and generating recommendations..."):
#             try:
#                 # Use Syracuse, NY if no input provided
#                 location = city_input if city_input.strip() else "Syracuse, NY"
                
#                 # Get recommendation
#                 recommendation = get_clothing_recommendation(location)
                
#                 # Store in session state
#                 st.session_state.recommendations.insert(0, {
#                     'location': location,
#                     'recommendation': recommendation
#                 })
                
#                 # Keep only last 5 recommendations
#                 st.session_state.recommendations = st.session_state.recommendations[:5]
                
#             except Exception as e:
#                 st.error(f"An error occurred: {str(e)}")
    
#     # Display current recommendation
#     if st.session_state.recommendations:
#         current = st.session_state.recommendations[0]
        
#         st.markdown("---")
#         st.subheader(f"📍 Clothing Recommendations for {current['location']}")
        
#         # Display recommendation in a nice container
#         with st.container():
#             st.markdown(current['recommendation'])
        
#         # Show history if there are previous searches
#         if len(st.session_state.recommendations) > 1:
#             st.markdown("---")
#             with st.expander("📜 Previous Searches"):
#                 for i, rec in enumerate(st.session_state.recommendations[1:], 1):
#                     st.markdown(f"**{rec['location']}**")
#                     st.markdown(rec['recommendation'][:200] + "...")
#                     st.markdown("")

#     # Footer
#     st.markdown("---")
#     st.markdown("*Powered by OpenAI and OpenWeatherMap API*")

# if __name__ == "__main__":
#     main()

# import requests
# import streamlit as st
# from openai import OpenAI
# import json



# # -------------------------------
# # Weather Function
# # -------------------------------
# def get_current_weather(location: str, api_key: str):
#     """Fetch current weather data from OpenWeatherMap API"""
#     if "," in location:
#         location = location.split(",")[0].strip()
#     url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}"
#     response = requests.get(url)
#     if response.status_code != 200:
#         return None
#     data = response.json()
#     try:
#         # Kelvin → Celsius
#         temp = data["main"]["temp"] - 273.15
#         feels_like = data["main"]["feels_like"] - 273.15
#         description = data["weather"][0]["description"]
#         return {
#             "location": location,
#             "temperature": round(temp, 1),
#             "feels_like": round(feels_like, 1),
#             "description": description,
#         }
#     except KeyError:
#         return None

# # -------------------------------
# # API Keys
# # -------------------------------
# OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
# WEATHER_API_KEY = st.secrets["Openweather_API_key"]

# # -------------------------------
# # Streamlit UI
# # -------------------------------
# city = st.text_input("Enter a city:", "Syracuse, NY")

# if st.button("Get Travel Weather Suggestion"):
#     # Step 1: Retrieve weather (always required)
#     if not city.strip():
#         city = "Syracuse, NY"
    
#     weather = get_current_weather(city, WEATHER_API_KEY)
    
#     if not weather:
#         st.error("Could not fetch weather data. Try another city.")
#     else:
#         # Step 2: Build weather context
#         weather_context = (
#             f"The weather in {weather['location']} is {weather['description']}, "
#             f"temperature {weather['temperature']}°C, feels like {weather['feels_like']}°C."
#         )
        
#         # Display weather information
#         st.subheader(f"Weather in {weather['location']}")
#         st.write(f"☁️ {weather['description']}")
#         st.write(f"🌡 {weather['temperature']}°C (feels like {weather['feels_like']}°C)")
        
#         # Step 3: Generate clothing + picnic suggestion using GPT-4o
#         client = OpenAI(api_key=OPENAI_API_KEY)
#         suggestion = client.chat.completions.create(
#             model="gpt-4o",
#             messages=[
#                 {"role": "system", "content": "You are a Expert travel assistant that suggests clothing and picnic advice based on current weather."},
#                 {"role": "user", "content": weather_context},
#                 {"role": "user", "content": "What should I wear today? "}
#             ],
#             temperature=0.2,
#             max_tokens=200,
#         )
#         advice = suggestion.choices[0].message.content
        
#         # Display suggestion
#         st.subheader("👕 Travel Suggestion")
#         st.write(advice)

import requests
import streamlit as st
from openai import OpenAI
import json

# Weather Function

def get_current_weather(location: str, api_key: str):
    """Fetch current weather data from OpenWeatherMap API"""
    if "," in location:
        location = location.split(",")[0].strip()
    url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}"
    response = requests.get(url)
    if response.status_code != 200:
        return None
    data = response.json()
    try:
        # Kelvin → Celsius
        temp = data["main"]["temp"] - 273.15
        feels_like = data["main"]["feels_like"] - 273.15
        description = data["weather"][0]["description"]
        return {
            "location": location,
            "temperature": round(temp, 1),
            "feels_like": round(feels_like, 1),
            "description": description,
        }
    except KeyError:
        return None


# API Keys
# -------------------------------
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
WEATHER_API_KEY = st.secrets["Openweather_API_key"]

# Streamlit UI

city = st.text_input("Enter a city:", "Syracuse, NY")

if st.button("Get Travel Weather Suggestion"):
    # Step 1: Retrieve weather (always required)
    if not city.strip():
        city = "Syracuse, NY"
    
    weather = get_current_weather(city, WEATHER_API_KEY)
    
    if not weather:
        st.error("Could not fetch weather data. Try another city.")
    else:
        # Step 2: Build weather context
        weather_context = (
            f"The weather in {weather['location']} is {weather['description']}, "
            f"temperature {weather['temperature']}°C, feels like {weather['feels_like']}°C."
        )
        
        # Display weather information
        st.subheader(f"Weather in {weather['location']}")
        st.write(f" {weather['description']}")
        st.write(f" {weather['temperature']}°C (feels like {weather['feels_like']}°C)")
        
        # Step 3: Generate clothing + picnic suggestion using GPT-4o
        client = OpenAI(api_key=OPENAI_API_KEY)
        suggestion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a Expert travel assistant that suggests clothing and picnic advice based on current weather."},
                {"role": "user", "content": weather_context},
                {"role": "user", "content": "What should I wear today? "}
            ],
            temperature=0.2,
            max_tokens=200,
        )
        advice = suggestion.choices[0].message.content
        
        # Display suggestion
        st.subheader("👕 Travel Suggestion")
        st.write(advice)
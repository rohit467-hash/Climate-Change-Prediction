import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# --- 1. YOUR NOTEBOOK FUNCTIONS (Integrated) ---
def get_current_weather(city):
    API_KEY = "be45314bce4008639d5a2ae0e56f4ba2" 
    BASE_URL = "https://api.openweathermap.org/data/2.5/"
    url = f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code != 200:
        return None
    data = response.json()
    return {
        'temp_min': data['main']['temp_min'],
        'temp_max': data['main']['temp_max'],
        'humidity': data['main']['humidity'],
        'pressure': data['main']['pressure'],
        'current_temp': data['main']['temp'],
        'WindGustSpeed': data['wind'].get('gust', data['wind']['speed']),
        'description': data['weather'][0]['description']
    }

# --- 2. THE GUI CLASS ---
class WeatherApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ML Weather Predictor")
        self.root.geometry("400x300")
        
        # Load and Train Model immediately
        self.setup_ml()
        self.create_widgets()

    def setup_ml(self):
        try:
            # Preparing data exactly like your notebook
            df = pd.read_csv("weather.csv").dropna()
            le = LabelEncoder()
            df['WindGustDir'] = le.fit_transform(df['WindGustDir'])
            df['RainTomorrow'] = le.fit_transform(df['RainTomorrow'])
            
            X = df[['MinTemp', 'MaxTemp', 'WindGustDir', 'WindGustSpeed', 'Humidity', 'Pressure', 'Temp']]
            y = df['RainTomorrow']
            
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X, y)
        except Exception as e:
            messagebox.showerror("Model Error", f"Could not train model: {e}")

    def create_widgets(self):
        tk.Label(self.root, text="Enter City Name:").pack(pady=10)
        self.city_entry = tk.Entry(self.root)
        self.city_entry.pack(pady=5)
        
        tk.Button(self.root, text="Predict Rain", command=self.on_predict_click).pack(pady=20)
        
        self.result_label = tk.Label(self.root, text="", font=("Helvetica", 12, "bold"))
        self.result_label.pack(pady=10)

    def on_predict_click(self):
        city = self.city_entry.get()
        data = get_current_weather(city)
        
        if data:
            # Create the input features matching your model's 7 columns
            input_df = pd.DataFrame([{
                'MinTemp': data['temp_min'],
                'MaxTemp': data['temp_max'],
                'WindGustDir': 0, # Placeholder for encoded direction
                'WindGustSpeed': data['WindGustSpeed'],
                'Humidity': data['humidity'],
                'Pressure': data['pressure'],
                'Temp': data['current_temp']
            }])
            
            prediction = self.model.predict(input_df)[0]
            status = "Rain Likely 🌧️" if prediction == 1 else "No Rain ☀️"
            color = "blue" if prediction == 1 else "green"
            
            self.result_label.config(text=f"Prediction: {status}\n({data['description']})", fg=color)
        else:
            messagebox.showerror("Error", "City not found!")

# --- 3. RUN THE APPLICATION ---
if __name__ == "__main__":
    root = tk.Tk()
    app = WeatherApp(root)
    root.mainloop()
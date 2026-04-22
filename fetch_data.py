import requests
import pandas as pd
import time

API_URL = "http://:8000/predict_grid"

while True:
    try:
        response = requests.get(API_URL)
        data = response.json()

        df = pd.DataFrame(data)

        df.to_csv("flood_data.csv", index=False)
        print("Updated data!")

    except Exception as e:
        print("Error:", e)

    time.sleep(30)  # refresh every 30 seconds

import requests
import pandas as pd
from datetime import datetime, timedelta
import time

# === CONFIGURATION ===
API_KEY = "DVRA7C4SU9VR9KUB56ENKFKJX"
# 🔁 Replace with your Visual Crossing API key
LOCATION = "New York"
START_DATE = "2023-01-01"
END_DATE = "2023-01-30"
OUTPUT_FILE = "weather_data.csv"


# === FETCH WEATHER DATA FUNCTION ===
def fetch_weather_data(location, start_date, end_date, api_key):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    all_data = []

    while start <= end:
        date_str = start.strftime("%Y-%m-%d")
        url = (
            "https://weather.visualcrossing.com/VisualCrossingWebServices"
            "/rest/"
            f"services/timeline/{location}/{date_str}"
            f"?unitGroup=metric&key={api_key}&include=hours"
        )

        print(f"Fetching weather for {date_str}...")
        try:
            response = requests.get(url)
            response.raise_for_status()
            json_data = response.json()

            if (
                'days' in json_data
                and len(json_data['days']) > 0
                and 'hours' in json_data['days'][0]
            ):
                for hour in json_data['days'][0]['hours']:
                    hour['date'] = date_str
                    all_data.append(hour)
            else:
                print(f"No hourly data found for {date_str}.")

        except Exception as e:
            print(f"Error fetching data for {date_str}: {e}")

        time.sleep(1)  # 💤 Be polite to the API
        start += timedelta(days=1)

    return pd.DataFrame(all_data)


# === MAIN EXECUTION ===
if __name__ == "__main__":
    weather_df = fetch_weather_data(
        LOCATION, START_DATE, END_DATE, API_KEY
    )

    if not weather_df.empty:
        weather_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n Weather data saved to '{OUTPUT_FILE}'")
        print(
            weather_df[
                ['datetime', 'temp', 'precip', 'conditions', 'date']
            ].head()
        )
    else:
        print("No data fetched.")

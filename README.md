# NYC Taxi Fare Dashboard

An interactive web application that visualizes NYC taxi fare data and provides fare predictions based on various factors including weather conditions.

## Features

- 📊 Interactive visualizations of taxi fare data
- 🌤️ Weather impact analysis on taxi fares
- 🔮 Fare prediction model
- 📈 Multiple data visualization charts
- 🔍 Advanced filtering options
- 💾 Data export functionality

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd RidePulse
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Data Files

The application requires the following data files:
- `nyc_taxi_weather.csv`: Main dataset containing taxi and weather information
- `nyc_taxi_filtered.csv`: Filtered dataset for analysis
- `nyc_weather_data.zip`: Weather data archive
- `NYC_taxi_fare_data.zip`: Taxi fare data archive

## Running the Application

To start the application, run:
```bash
streamlit run taxi_app.py
```

The application will open in your default web browser at `http://localhost:8501`.

## Usage

### Visualizations Tab
- Use the sidebar filters to customize the data view
- Explore various charts and metrics
- Download filtered data for further analysis

### Prediction Tab
- Input trip details (passengers, distance, time, etc.)
- Get instant fare predictions
- Download prediction results

## Project Structure

```
RidePulse/
├── taxi_app.py          # Main application file
├── requirements.txt     # Project dependencies
├── assets/             # Static assets
├── nyc/                # NYC-specific data
└── README.md          # Project documentation
```

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
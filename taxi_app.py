import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
import plotly.graph_objects as go

# ---------------- CONFIG ----------------
st.set_page_config(page_title="NYC Taxi Fare Dashboard", layout="wide")

st.markdown("""
<style>
body { background-color: #000000; color: white; }
h1, h2, h3 { color: #f5b700; }
.sidebar .sidebar-content { background-color: #111111; }
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("nyc_taxi_weather.csv", parse_dates=["pickup_datetime"])

df = load_data()

# ---------------- NAVIGATION ----------------
tab1, tab2 = st.columns([1, 1])
with tab1:
    if st.button("📊 Visualizations"):
        st.session_state.view = "Visualizations"
with tab2:
    if st.button("🔮 Prediction"):
        st.session_state.view = "Prediction"
if "view" not in st.session_state:
    st.session_state.view = "Visualizations"

# ---------------- VISUALIZATION TAB ----------------
if st.session_state.view == "Visualizations":
    st.title("📊 NYC Taxi Fare & Weather Dashboard")

    st.sidebar.header("🔧 Filters")
    passenger = st.sidebar.slider("Passenger Count", 1, 6, 1)
    fare_range = st.sidebar.slider("Fare ($)", 0.0, 100.0, (5.0, 50.0))
    dist_range = st.sidebar.slider("Distance (km)", 0.0, 30.0, (0.0, 10.0))
    temp_range = st.sidebar.slider("Temperature (°C)", -10.0, 40.0, (-5.0, 30.0))
    rain_range = st.sidebar.slider("Precipitation (mm)", 0.0, 50.0, (0.0, 5.0))

    filtered = df[
        (df["passenger_count"] == passenger) &
        (df["fare_amount"].between(*fare_range)) &
        (df["distance_km"].between(*dist_range)) &
        (df["temperature_2m"].between(*temp_range)) &
        (df["precipitation"].between(*rain_range))
    ]

    # KPIs
    st.metric("Total Rides", f"{len(filtered):,}")
    st.metric("Avg Fare", f"${filtered['fare_amount'].mean():.2f}")
    st.metric("Avg Temp", f"{filtered['temperature_2m'].mean():.1f} °C")

    st.download_button("⬇ Download Filtered Data", data=filtered.to_csv(index=False), file_name="filtered_data.csv")

    # Charts
    st.plotly_chart(px.histogram(filtered, x="fare_amount", nbins=40, title="Fare Distribution", color_discrete_sequence=["#f5b700"]), use_container_width=True)
    st.plotly_chart(px.line(filtered.groupby("hour")["fare_amount"].mean().reset_index(), x="hour", y="fare_amount", title="Avg Fare by Hour", markers=True, color_discrete_sequence=["#f5b700"]), use_container_width=True)
    st.plotly_chart(px.bar(df.groupby("day_of_week")["fare_amount"].mean().reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]).reset_index(), x="day_of_week", y="fare_amount", title="Avg Fare by Day", color_discrete_sequence=["#f5b700"]), use_container_width=True)

    # NEW WEATHER VISUALS
    st.plotly_chart(px.line(filtered, x="temperature_2m", y="fare_amount", title="Fare vs Temperature (°C)", color_discrete_sequence=["#ffcc00"]), use_container_width=True)
    st.plotly_chart(px.scatter(filtered, x="windspeed_10m", y="fare_amount", title="Fare vs Wind Speed (km/h)", opacity=0.5, color_discrete_sequence=["#f5b700"]), use_container_width=True)
    st.plotly_chart(px.bar(filtered, x="precipitation", y="fare_amount", title="Fare vs Precipitation (mm)", color_discrete_sequence=["#f39c12"]), use_container_width=True)

    st.plotly_chart(px.box(filtered, x="passenger_count", y="fare_amount", title="Fare by Passenger Count", color_discrete_sequence=["#f5b700"]), use_container_width=True)

    with st.expander("📄 View Data"):
        st.dataframe(filtered.head(100))

# ---------------- PREDICTION TAB ----------------
else:
    st.title("🔮 Taxi Fare Prediction")

    user_passenger = st.slider("Passenger Count", 1, 6, 1)
    user_distance = st.slider("Trip Distance (km)", 0.1, 30.0, 3.0)
    user_hour = st.slider("Pickup Hour", 0, 23, 14)
    user_day = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    user_temp = st.slider("Temperature (°C)", -10.0, 40.0, 22.0)
    user_rain = st.slider("Precipitation (mm)", 0.0, 50.0, 1.0)

    df_model = df[["passenger_count", "distance_km", "hour", "day_of_week", "temperature_2m", "precipitation", "fare_amount"]].dropna()

    enc = OneHotEncoder(sparse_output=False)
    X_encoded = enc.fit_transform(df_model[["day_of_week"]])
    encoded_cols = enc.get_feature_names_out(["day_of_week"])

    X = pd.concat([
        df_model.drop(columns=["fare_amount", "day_of_week"]).reset_index(drop=True),
        pd.DataFrame(X_encoded, columns=encoded_cols)
    ], axis=1)
    y = df_model["fare_amount"]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    user_encoded = enc.transform([[user_day]])
    user_features = np.hstack([[user_passenger, user_distance, user_hour, user_temp, user_rain], user_encoded[0]])
    fare_pred = model.predict([user_features])[0]

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=fare_pred,
        title={"text": "Estimated Fare ($)", "font": {"size": 24}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#f5b700"},
            "steps": [
                {"range": [0, 30], "color": "#d2f5a7"},
                {"range": [30, 60], "color": "#ffe36e"},
                {"range": [60, 100], "color": "#f5a45d"},
            ]
        }
    ))
    st.plotly_chart(gauge, use_container_width=True)

    st.download_button("⬇ Download Prediction", data=pd.DataFrame([{
        "passenger_count": user_passenger,
        "distance_km": user_distance,
        "hour": user_hour,
        "day_of_week": user_day,
        "temperature_2m": user_temp,
        "precipitation": user_rain,
        "predicted_fare": fare_pred
    }]).to_csv(index=False), file_name="predicted_fare.csv")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets from EDA directory
weather = pd.read_csv('EDA/weather.csv')
events = pd.read_csv('EDA/events.csv')

# --- 1. Clean & Process Weather Data ---
weather.columns = ["area_name", "date", "max_temp_f", "min_temp_f", "precipitation_inches"]
weather["date"] = pd.to_datetime(weather["date"].astype(str), format="%Y%m%d", errors="coerce")
weather["average_temp_f"] = (weather["max_temp_f"] + weather["min_temp_f"]) / 2
weather["is_raining"] = weather["precipitation_inches"].fillna(0).gt(0)

# Filter invalid entries and extract time variables
weather_clean = weather[weather["max_temp_f"] > 0].copy()
weather_clean['month'] = weather_clean['date'].dt.month

# Calculate total number of unique years to scale rainy days to an annual average
num_years = weather_clean['date'].dt.year.nunique()

monthly_weather = weather_clean.groupby('month').agg(
    avg_temp=('average_temp_f', 'mean'),
    avg_rainy_days=('is_raining', lambda x: x.sum() / num_years)  # Averaged per year
).reset_index()

# --- 2. Clean & Process Events Data ---
events = events.rename(columns={
    "Event ID": "event_id", "Event Class": "event_class",
    "Effective From Date": "start_date", "Effective To Date": "end_date"
})
events["start_date"] = pd.to_datetime(events["start_date"], errors="coerce")
events["end_date"] = pd.to_datetime(events["end_date"], errors="coerce").fillna(events["start_date"])
events['duration_days'] = (events['end_date'] - events['start_date']).dt.days + 1

expanded_rows = []
for event in events.itertuples(index=False):
    if pd.isna(event.start_date) or pd.isna(event.end_date):
        continue
    for date in pd.date_range(event.start_date, event.end_date, freq="D"):
        expanded_rows.append({"date": date, "event_id": event.event_id, "event_class": event.event_class})

expanded = pd.DataFrame(expanded_rows)
daily_events = expanded.groupby("date", as_index=False).agg(event_count=("event_id", "nunique"))

sns.set_theme(style="whitegrid")

# =========================================================================
# SLIDE 1: Weather Data Analysis (3 Graphs Side-by-Side)
# =========================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Graph 1.1: Temperature Distributions
sns.kdeplot(data=weather_clean["average_temp_f"], ax=axes[0], fill=True, color="crimson", label="Avg Temp (°F)")
sns.kdeplot(data=weather_clean["max_temp_f"], ax=axes[0], fill=True, color="orange", label="Max Temp (°F)")
sns.kdeplot(data=weather_clean["min_temp_f"], ax=axes[0], fill=True, color="skyblue", label="Min Temp (°F)")
axes[0].set_title("1. Temperature Distributions", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Temperature (°F)")
axes[0].set_ylabel("Density")
axes[0].legend()

# Graph 1.2: Rain Occurrence Rate
rain_counts = weather_clean["is_raining"].value_counts(normalize=True) * 100
axes[1].bar(["No Rain (80.7%)", "Rain (19.3%)"], rain_counts, color=["#a8dadc", "#457b9d"], width=0.4)
axes[1].set_title("2. Daily Precipitation Rate", fontsize=12, fontweight="bold")
axes[1].set_ylabel("Percentage of Days (%)")

# Graph 1.3: Monthly Climate Trends (Corrected Scale)
ax_twin = axes[2].twinx()
axes[2].plot(monthly_weather['month'], monthly_weather['avg_temp'], color='crimson', marker='o', linewidth=2)
ax_twin.bar(monthly_weather['month'], monthly_weather['avg_rainy_days'], color='#457b9d', alpha=0.4, width=0.5)

axes[2].set_title("3. Monthly Climate Trends (Avg per Year)", fontsize=12, fontweight="bold")
axes[2].set_xlabel("Month")
axes[2].set_ylabel("Avg Temp (°F)", color="crimson")
ax_twin.set_ylabel("Avg Rainy Days / Month", color="#457b9d")

plt.tight_layout()
plt.savefig("slide1_weather_analysis.png", dpi=300)
plt.close()

# =========================================================================
# SLIDE 2: City Events Analysis (3 Graphs Side-by-Side)
# =========================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Graph 2.1: Event Distribution by Class
class_counts = events["event_class"].value_counts()
sns.barplot(x=class_counts.values, y=class_counts.index, ax=axes[0], palette="Blues_r")
axes[0].set_title("1. Event Distribution by Class", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Total Event Count")
axes[0].set_ylabel("Event Class")

# Graph 2.2: Concurrent Active Events per Day
sns.countplot(data=daily_events, x="event_count", ax=axes[1], palette="crest")
axes[1].set_title("2. Concurrent Events per Day", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Number of Concurrent Events")
axes[1].set_ylabel("Number of Days")

# Graph 2.3: Event Duration Distribution
sns.histplot(events['duration_days'], bins=15, color='#2a9d8f', kde=False, ax=axes[2])
axes[2].set_title("3. Event Duration Distribution", fontsize=12, fontweight="bold")
axes[2].set_xlabel("Duration (Days)")
axes[2].set_ylabel("Event Count")

plt.tight_layout()
plt.savefig("slide2_events_analysis.png", dpi=300)
plt.close()
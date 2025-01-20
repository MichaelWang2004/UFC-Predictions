import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Helper function to convert height/reach to centimeters
def height_to_cm(value):
    try:
        if "'" in value:  # Handle height in format 6'1"
            parts = value.split("'")
            feet = int(parts[0].strip())
            inches = int(parts[1].replace('"', '').strip())
            return round((feet * 30.48) + (inches * 2.54),2)
        elif '"' in value:  # Handle reach in format 72"
            inches = int(value.replace('"', '').strip())
            return round(inches * 2.54,2)
        return np.nan  # Return NaN for invalid formats
    except Exception:
        return np.nan

# Load the data
data = pd.read_csv("ufc_combined.csv")

# Preprocessing: Remove rows with missing or irrelevant data
data.dropna(inplace=True)

# Convert height and reach columns to centimeters
data["height_fighter1"] = data["height_fighter1"].apply(height_to_cm)
data["height_fighter2"] = data["height_fighter2"].apply(height_to_cm)
data["reach_fighter1"] = data["reach_fighter1"].apply(height_to_cm)
data["reach_fighter2"] = data["reach_fighter2"].apply(height_to_cm)

# Preprocessing: Remove rows with missing or irrelevant data
data.dropna(inplace=True)

# Remove 'Catchweight' fights
data = data[data["weight_class"] != "Catch Weight"]

# Convert categorical features to numerical (example for stance and weight_class)
weight_class_mapping = {
    'Strawweight': 115, 'Flyweight': 125, 'Bantamweight': 135, 'Featherweight': 145,
    'Lightweight': 155, 'Welterweight': 170, 'Middleweight': 185,
    'Light Heavyweight': 205, 'Heavyweight': 265
}
data["weight_class"] = data["weight_class"].map(weight_class_mapping)

data["stance_fighter1"] = data["stance_fighter1"].apply(lambda x: 1 if x == "Orthodox" else 0)
data["stance_fighter2"] = data["stance_fighter2"].apply(lambda x: 1 if x == "Orthodox" else 0)

# write the filtered data to a new csv file
data.to_csv("ufc_combined_modified.csv", index=False)

# Define features and target variable
features = [
    'weight_class', 'height_fighter1', 'reach_fighter1', 'stance_fighter1',
    'SLpM_fighter1', 'Str_Acc_fighter1', 'SApM_fighter1', 'Str_Def_fighter1',
    'TD_Avg_fighter1', 'TD_Acc_fighter1', 'TD_Def_fighter1', 'Sub_Avg_fighter1',
    'win%_fighter1', 'height_fighter2', 'reach_fighter2', 'stance_fighter2',
    'SLpM_fighter2', 'Str_Acc_fighter2', 'SApM_fighter2', 'Str_Def_fighter2',
    'TD_Avg_fighter2', 'TD_Acc_fighter2', 'TD_Def_fighter2', 'Sub_Avg_fighter2',
    'win%_fighter2'
]
X = data[features]
y = data["winner"]  # 1 if fighter1 wins, 0 if fighter2 wins

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the data
data = pd.read_csv("ufc_combined.csv")

# Preprocessing: Remove rows with missing or irrelevant data
data.dropna(inplace=True)

# Remove 'Catchweight' fights as no weight class is specified, making the data unhelpfuk
data = data[data["weight_class"] != "Catch Weight"]

# Convert data to numerical values (weight class, stance)
weight_class_mapping = {
    'Strawweight': 1, 'Flyweight': 2, 'Bantamweight': 3, 'Featherweight': 4,
    'Lightweight': 5, 'Welterweight': 6, 'Middleweight': 7,
    'Light Heavyweight': 8, 'Heavyweight': 9
}
data["weight_class"] = data["weight_class"].map(weight_class_mapping)

data["stance_fighter1"] = data["stance_fighter1"].apply(lambda x: 1 if x == "Orthodox" else 0)
data["stance_fighter2"] = data["stance_fighter2"].apply(lambda x: 1 if x == "Orthodox" else 0)

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

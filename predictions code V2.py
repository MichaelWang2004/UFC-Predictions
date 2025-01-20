import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_data(data: pd.DataFrame) -> tuple[RandomForestClassifier]:
    """
    Trains a Random Forest Classifier on the provided dataset containing statistics about
    ufc fighters and evaluates its performance.

    Inputs:
        data (pd.DataFrame): A pandas DataFrame containing features and the target variable (winner).
        This data is obtained from "ufc_combined_modified.csv"

    Outputs:
        tuple: A tuple containing the trained random forest classifier
    """

    # Define features and target variable.
    # Non-important data is omited from the features.
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

    # Evaluate the model's performance and print to the console
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))
    
    return model

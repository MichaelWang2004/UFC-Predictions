import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

## Data Source
#The data used in this project was retrieved from the following sources:
#1. https://github.com/naity/DeepUFC2

def train_data(data: pd.DataFrame) -> tuple[RandomForestClassifier, list]:
    """
    Trains a Random Forest Classifier on the provided dataset containing statistics about
    ufc fighters and evaluates its performance.

    Inputs:
        data: A pandas DataFrame containing features and the target variable (winner).
        this data is obtained from "ufc_combined_modified.csv"

    Outputs:
        tuple: A tuple containing:
            - model: The trained Random Forest model.
            - features: The list of feature names used for training.

    Complexity:
        I'm not sure how to do complexity for a machine learning model
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
    
    return model,features

def get_fighter_stats(name:str, fighters_data: pd.DataFrame):
    """
    obtains a row from the dataset which has a name attribute which is the same as the target fighter
    Inputs:
        name: Name of fighter to find info about
    Outputs:
        fighters_data: The CSV file which contains information about fighters ("ufc_fighters_modified")
    Complexity:
        O(N) where N is the number of rows as we have to look through each row and check the name.

    """

    fighter_row = fighters_data[fighters_data['name'] == name]
    if fighter_row.empty:
        print(f"Fighter '{name}' not found in the dataset.")
        return None
    # Get the first matching row. Raises error if I dont
    fighter_row = fighter_row.iloc[0]

    return fighter_row

def predict_fight_winner(fighter1_name:str, fighter2_name:str, model, fighters_data:pd.DataFrame, features:list[str])->None:
    """
    Predicts the outcome of a fight between two fighters and outputs the likelihood of each fighter winning.

    Inputs:
        fighter1_name: The name of the first fighter.
        fighter2_name: The name of the second fighter.
        model: The trained machine learning model
        fighters_data: A pandas DataFrame containing fighter statistics.
        features: A list of feature names to match the model's input format.

    Outputs:
        None: Prints the likelihood of each fighter winning the fight.

    Complexity:
        I'm not sure how to do complexity for a machine learning model
    """

    # Get stats for both fighters
    fighter1_stats = get_fighter_stats(fighter1_name, fighters_data)
    fighter2_stats = get_fighter_stats(fighter2_name, fighters_data)

    if fighter1_stats is None or fighter2_stats is None:
        return  # Exit if any fighter's data is missing

    # Combine fighter stats into a single feature row
    fighter_features = [
        1,  # Assume same weight class, (weight class should really be removed from features)
        fighter1_stats["height"], fighter1_stats["reach"], fighter1_stats["stance"],
        fighter1_stats["SLpM"], fighter1_stats["Str_Acc"], fighter1_stats["SApM"], fighter1_stats["Str_Def"],
        fighter1_stats["TD_Avg"], fighter1_stats["TD_Acc"], fighter1_stats["TD_Def"], fighter1_stats["Sub_Avg"],
        fighter1_stats["win%"], fighter2_stats["height"], fighter2_stats["reach"], fighter2_stats["stance"],
        fighter2_stats["SLpM"], fighter2_stats["Str_Acc"], fighter2_stats["SApM"], fighter2_stats["Str_Def"],
        fighter2_stats["TD_Avg"], fighter2_stats["TD_Acc"], fighter2_stats["TD_Def"], fighter2_stats["Sub_Avg"],
        fighter2_stats["win%"]
    ]

    # convert to a dataframe so that it can be inputted into the model.
    fighter_features_df = pd.DataFrame([fighter_features], columns=features)
    # Obtain the probabilities of each fighter winning
    probabilities = model.predict_proba(fighter_features_df)[0]

    # Print results
    print(f"Prediction for {fighter1_name} vs {fighter2_name}:")
    print(f"Likelihood of {fighter1_name} winning: {probabilities[1]:.2f}")
    print(f"Likelihood of {fighter2_name} winning: {probabilities[0]:.2f}")

def main():
    """
    The main function loads data from the csv files "ufc_combined_modified.csv" and "ufc_fighters_modified.csv".
    It also itiates the model training and asks the user for fighters, allowing users to see predicted fight outcomes.

    Inputs:
        None
    Outputs:
        None
    Complexity:
        I'm not sure how to do complexity for a machine learning model
    """

    # Load the datasets
    data = pd.read_csv("ufc_combined_modified.csv")
    fighters_data = pd.read_csv("ufc_fighters_modified.csv")
    
    # Train the model
    print("Training the model...")
    model,features = train_data(data)
    print("Model training complete!")
    
    # Allow user to input fighter names for predictions
    while True:
        print("\Please note that the dataset only contains data up to 2018")
        print("\nEnter the names of the two fighters (or type 'exit' to quit):")
        fighter1_name = input("Enter Fighter 1's name: ").strip()

        #if exit is typed, the code quits
        if fighter1_name.lower() == "exit":
            break
        fighter2_name = input("Enter Fighter 2's name: ").strip()
        if fighter2_name.lower() == "exit":
            break

        # Make predictions
        try:
            predict_fight_winner(fighter1_name, fighter2_name, model, fighters_data, features)
        except Exception as e:
            print(f"An error occurred during prediction: {e}")

# Run the main function
if __name__ == "__main__":
    main()

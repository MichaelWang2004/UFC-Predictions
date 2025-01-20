import pandas as pd
import numpy as np

def height_to_cm(value: str) -> float:
    """
    Converts a height or reach value from feet and or inches format to centimeters.

    Inputs:
        value (str): A string representing height or reach in feet and or inches.

    Outputs:
        float: The converted value in centimeters, rounded to two decimal places.
        If the input format is invalid, returns NaN.
    """

    try:
        # For the instance where the height is in feet and inches
        if "'" in value: 
            parts = value.split("'")
            feet = int(parts[0].strip())
            inches = int(parts[1].replace('"', '').strip())
            return round((feet * 30.48) + (inches * 2.54),2)
        
        #For converting reach which is only in inches
        elif '"' in value:
            inches = int(value.replace('"', '').strip())
            return round(inches * 2.54,2)
        return np.nan  # Return NaN for invalid formats
    except Exception:
        return np.nan
    
    
def process_data() -> None:
    """
    Processes the UFC fight data by cleaning, transforming, and saving it to a new CSV file.
    Inputs:
        None

    Outputs:
        None
    """

    # Load the data
    data = pd.read_csv("ufc_combined.csv")

    # Preprocessing: Remove rows with missing or irrelevant data
    data.dropna(inplace=True)

    # Convert height and reach columns to centimeters using the function defined above
    data["height_fighter1"] = data["height_fighter1"].apply(height_to_cm)
    data["height_fighter2"] = data["height_fighter2"].apply(height_to_cm)
    data["reach_fighter1"] = data["reach_fighter1"].apply(height_to_cm)
    data["reach_fighter2"] = data["reach_fighter2"].apply(height_to_cm)

    # Remove 'Catchweight' fights as they don't provide any information about a fighter's weight
    # Thus it is not useful to us for our model
    data = data[data["weight_class"] != "Catch Weight"]

    #Convert values for stance and weight class into numerical values so they can actually be used.
    weight_class_mapping = {
        'Strawweight': 1, 'Flyweight': 2, 'Bantamweight': 3, 'Featherweight': 4,
        'Lightweight': 5, 'Welterweight': 6, 'Middleweight': 7,
        'Light Heavyweight': 8, 'Heavyweight': 9
    }
    data["weight_class"] = data["weight_class"].map(weight_class_mapping)

    data["stance_fighter1"] = data["stance_fighter1"].apply(lambda x: 1 if x == "Orthodox" else 0)
    data["stance_fighter2"] = data["stance_fighter2"].apply(lambda x: 1 if x == "Orthodox" else 0)

    # write the filtered data to a new csv file
    data.to_csv("ufc_combined_modified.csv", index=False)



def process_fighters_data() -> None:
    """
    Processes the UFC fighter data by cleaning, transforming, and saving it to a new CSV file.
    Inputs:
        None

    Outputs:
        None
    """

    fighters_data = pd.read_csv("ufc_fighters.csv")
    # Convert height and reach to cm
    fighters_data['height'] = fighters_data['height'].apply(height_to_cm)
    fighters_data['reach'] = fighters_data['reach'].apply(height_to_cm)

    # Convert percentage columns to decimal
    fighters_data['Str_Acc'] = fighters_data['Str_Acc'].str.strip("%")
    fighters_data['Str_Def'] = fighters_data['Str_Def'].str.strip("%")
    fighters_data['TD_Acc'] = fighters_data['TD_Acc'].str.strip("%")
    fighters_data['TD_Def'] = fighters_data['TD_Def'].str.strip("%")

    # Convert stance to numerical
    fighters_data['stance'] = fighters_data['stance'].apply(lambda x: 1 if x == "Orthodox" else 0)

    # Drop rows with missing data
    fighters_data.dropna(inplace=True)

    # write the filtered data to a new csv file
    fighters_data.to_csv("ufc_fighters_modified.csv", index=False)

process_fighters_data()
process_data()
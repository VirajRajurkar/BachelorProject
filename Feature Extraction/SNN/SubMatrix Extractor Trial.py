import pandas as pd

# Load the CSV file while ignoring the last column
file_path = "C:/Users/MDilf/Desktop/Anka Datenbearbeitung/Feature Extraction/Feature_SNN/SNN_Binary/SNN_Unordered/SNN_Feature_Matrix_Ara.csv"  # Replace with the path to your CSV file
df = pd.read_csv(file_path).iloc[:, :-1]  # Ignore the last column

# Define the ranges
subject_ranges = {
    "No1": slice(0, 160),        # NAra
    "No2": slice(160, 320),     # NDan
    "No3": slice(320, 480),     # NEng
    "No4": slice(480, 640),    # NHin
    "No5": slice(640, 800),
    "No6": slice(800, 960),
    "No7": slice(960, 1120), 
    "No8": slice(1120, 1280), 
    "No9": slice(1280, 1440), 
    "No10": slice(1440, 1600), 
    "Yes1": slice(1600, 1760),  # YAra
    "Yes2": slice(1760, 1920),  # YDan
    "Yes3": slice(1920, 2080),  # YEng
    "Yes4": slice(2080, 2240),  # YHin
    "Yes5": slice(2240, 2400),  
    "Yes6": slice(2400, 2560),  # YAra
    "Yes7": slice(2560, 2720),  # YDan
    "Yes8": slice(2720, 2880),  # YEng
    "Yes9": slice(2880, 3040),  # YHin
    "Yes10": slice(3040, 3200), # YJap
}

# Split the data into slices based on the ranges
split_data = {key: df.iloc[range_] for key, range_ in subject_ranges.items()}

# Reorder the data
reorder_keys = ["No1", "Yes1", "No2", "Yes2", "No3", "Yes3", "No4", "Yes4", "No5", "Yes5", "No6", "Yes6", "No7", "Yes7", "No8", "Yes8", "No9", "Yes9", "No10", "Yes10"]
reordered_data = pd.concat([split_data[key] for key in reorder_keys], ignore_index=True)

# Add the new "Label" column
labels = (["No"] * 160 + ["Yes"] * 160) * 10
reordered_data["Label"] = labels

# Save the reordered data to a new CSV
output_path = "C:/Users/MDilf/Desktop/Anka Datenbearbeitung/Feature Extraction/Feature_SNN/SNN_Binary/SNN_Ordered_Trial/Reordered_SNN_Feature_Matrix_after_Trial_Ara.csv"  # Replace with the desired output file path
reordered_data.to_csv(output_path, index=False)

print(f"Reordered data saved to {output_path}")

import pandas as pd

# Load the CSV file while ignoring the last column
file_path = "C:/Users/MDilf/Desktop/Anka Datenbearbeitung/Feature Extraction/Feature_SNN/SNN_Binary/SNN_Unordered/SNN_Feature_Matrix_Jap.csv"  # Replace with the path to your CSV file
df = pd.read_csv(file_path).iloc[:, :-1]  # Ignore the last column

# Define the ranges
subject_ranges = {
    "No1": slice(0, 320),        # NAra
    "No2": slice(320, 640),     # NDan
    "No3": slice(640, 960),     # NEng
    "No4": slice(960, 1280),    # NHin
    "No5": slice(1280, 1600),   # NJap
    "Yes1": slice(1600, 1920),  # YAra
    "Yes2": slice(1920, 2240),  # YDan
    "Yes3": slice(2240, 2560),  # YEng
    "Yes4": slice(2560, 2880),  # YHin
    "Yes5": slice(2880, 3200)   # YJap
}

# Split the data into slices based on the ranges
split_data = {key: df.iloc[range_] for key, range_ in subject_ranges.items()}

# Reorder the data
reorder_keys = ["No1", "Yes1", "No2", "Yes2", "No3", "Yes3", "No4", "Yes4", "No5", "Yes5"]
reordered_data = pd.concat([split_data[key] for key in reorder_keys], ignore_index=True)

# Add the new "Label" column
labels = (["No"] * 320 + ["Yes"] * 320) * 5
reordered_data["Label"] = labels

# Save the reordered data to a new CSV
output_path = "C:/Users/MDilf/Desktop/Anka Datenbearbeitung/Feature Extraction/Feature_SNN/SNN_Binary/SNN_Ordered_Subject/Reordered_SNN_Feature_Matrix_after_Subject_Jap.csv"  # Replace with the desired output file path
reordered_data.to_csv(output_path, index=False)

print(f"Reordered data saved to {output_path}")

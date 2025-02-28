# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 09:51:53 2025

@author: shayj
"""

"Requires openpyxl"
import json
import pandas as pd
import numpy as np
import os

def write_json(Dataframe, Filepath, key_replacements=None):
    DF_json = Dataframe.to_json(orient="records")
    json_dict = json.loads(DF_json)
    
    if key_replacements:
        for i, item in enumerate(json_dict):
            for key, value in list(item.items()):
                if key in key_replacements:
                    json_dict[i][key_replacements[key]] = item.pop(key)
    
    with open(Filepath, "w", encoding="utf-8") as json_file:
        json.dump(json_dict, json_file, indent=4, default=str)

Test_Sheet_Names = False

# Load the Excel file
excel_file_path = "ISGS Well Completion Schema.xlsx"

excel_file = pd.ExcelFile(excel_file_path)

Trained_Models_df = pd.read_excel(excel_file_path, sheet_name="Trained Models")

sheet_names = excel_file.sheet_names
print(sheet_names)

Extractors = Trained_Models_df[(Trained_Models_df["Processor Type"] == "Extractor")*(Trained_Models_df["Primary Model in Processor"] == "primary")]
# print(f"extractors : {Extractors}")

if Test_Sheet_Names:
    for processor in Extractors["Processor Name"]:
        print(np.any(processor in sheet_names))

key_replacements = {
    "Google Processor Name": "google_processor_name",
    "OGRRE Document Type": "document_type",
    "Page Order Sort": "page_order_sort",
    "Name": "name",
    "Google Data type": "google_data_type",
    "Occurrence": "occurrence",
    "Grouping": "grouping",
    "Database Data Type": "database_data_type",
    "Cleaning Function": "cleaning_function",
    "Accepted Range": "accepted_range",
    "Field Specific Notes": "field_specific_notes",
    "Model Enabled": "model_enabled"
}

write_json(Dataframe = Extractors.reset_index()[Extractors.columns], Filepath = "Extractor Processors.json", key_replacements=key_replacements)

extractors_dir = "isgs_extractors"
if not os.path.exists(extractors_dir):
    os.makedirs(extractors_dir)

for processor in Extractors["Processor Name"]:
    processor_df = pd.read_excel(excel_file_path, sheet_name=processor)
    write_json(Dataframe =  processor_df, Filepath = f"isgs_extractors/{processor}.json", key_replacements=key_replacements)
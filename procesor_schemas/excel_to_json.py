# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 09:51:53 2025

@author: shayj
"""

"Requires openpyxl"
import json
import pandas as pd
import numpy as np

def write_json(Dataframe,Filepath):
    DF_json = Dataframe.to_json(orient='records')
    # Make the string into a list to be able to input in to a JSON-file
    thisisjson_dict = json.loads(DF_json)

    # Define file to write to and 'w' for write option -> json.dump() 
    # defining the list to write from and file to write to
    with open(Filepath, 'w', encoding="utf-8") as json_file:
        json.dump(thisisjson_dict, json_file, indent=4,default=str)

Test_Sheet_Names = False

# Load the Excel file
excel_file_path = 'ISGS Well Completion Schema.xlsx'

excel_file = pd.ExcelFile(excel_file_path)


Trained_Models_df = pd.read_excel(excel_file_path, sheet_name='Trained Models')
# Get the sheet names
sheet_names = excel_file.sheet_names
print(sheet_names)


Extractors = Trained_Models_df[(Trained_Models_df['Processor Type'] == 'Extractor')*(Trained_Models_df['Primary Model in Processor'] == 'primary')]

if Test_Sheet_Names:
    for processor in Extractors['Processor Name']:
        print(np.any(processor in sheet_names))

write_json(Dataframe = Extractors.reset_index()[Extractors.columns],Filepath = 'Extractor Processors.json')

for processor in Extractors['Processor Name']:
    processor_df = pd.read_excel(excel_file_path, sheet_name=processor)
    write_json(Dataframe =  processor_df,Filepath = '%s.json'%(processor))

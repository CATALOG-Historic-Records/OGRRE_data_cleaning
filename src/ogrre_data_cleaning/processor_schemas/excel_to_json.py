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

def Excel_to_Json(excel_file_path, Test_Sheet_Names = False, Organization = None):
    if isinstance(Organization,str):
        Organization = Organization
    else:
        index = max(excel_file_path.rfind(" Schema.xlsx"), excel_file_path.rfind(" schema.xlsx"))
        if index == -1:
            index = excel_file_path.rfind(".xlsx")
            
        Organization = excel_file_path[:index]
        
    Organization = Organization.lower().strip().replace(" ", "_")
        
    # Load the Excel file
    excel_file = pd.ExcelFile(excel_file_path)
    sheet_names = excel_file.sheet_names
    print(sheet_names)
    
    Trained_Models_df = pd.read_excel(excel_file_path, sheet_name="Trained Models")
    Extractors = Trained_Models_df[(Trained_Models_df["Processor Type"] == "Extractor")*(Trained_Models_df["Primary Model in Processor"] == "primary")]
    All_Splitters = Trained_Models_df[Trained_Models_df["Processor Type"] == "Splitter"]
    Splitters = All_Splitters[All_Splitters["Primary Model in Processor"] == "primary"].copy()
    if Splitters.empty and not All_Splitters.empty:
        Splitters = All_Splitters[All_Splitters["Model ID"].notna()].copy()

    if Test_Sheet_Names:
        for processor in Extractors["Processor Name"]:
            print(np.any(processor in sheet_names))
        for processor in Splitters["Processor Name"]:
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
        "Model Enabled": "model_enabled",
        "Alias": "alias"
    }
    processor_name_replacements = {
        "Osage_V1_Final_Report_Compl_Dep": "Osage_V1_Final_Report_Compl_Deep",
    } 
    
    extractors_dir = f"{Organization}_extractors"
    if not os.path.exists(extractors_dir):
        os.makedirs(extractors_dir)
    init_path = os.path.join(extractors_dir, "__init__.py")
    if not os.path.exists(init_path):
        with open(init_path, "w", encoding="utf-8") as f:
            pass  # Leave it empty
        f.close()
    for processor in processor_name_replacements:
        Extractors.loc[Extractors['Processor Name'] == processor,'Processor Name'] = processor_name_replacements[processor]
    Extractors = Extractors.astype({'Training Documents': 'float64','Testing Documents': 'float64'})
    write_json(Dataframe = Extractors.reset_index()[Extractors.columns], Filepath = f"{extractors_dir}/Extractor Processors.json", key_replacements=key_replacements)

    if not Splitters.empty:
        for processor in processor_name_replacements:
            Splitters.loc[Splitters['Processor Name'] == processor, 'Processor Name'] = processor_name_replacements[processor]
        Splitters = Splitters.astype({'Training Documents': 'float64', 'Testing Documents': 'float64'})
        splitter_records = json.loads(Splitters.reset_index()[Splitters.columns].to_json(orient="records"))
        for item in splitter_records:
            for key in list(item.keys()):
                if key in key_replacements:
                    item[key_replacements[key]] = item.pop(key)
            proc_name = item.get("Processor Name")
            sheet_key = None
            if proc_name in sheet_names:
                sheet_key = proc_name
            elif proc_name:
                sheet_key = next(
                    (
                        sh for sh in sheet_names
                        if sh == proc_name[:31] or sh.replace("_", "") == proc_name.replace("_", "")
                    ),
                    None,
                )
            if sheet_key:
                try:
                    splitter_df = pd.read_excel(excel_file_path, sheet_name=sheet_key)
                    if "Name" in splitter_df.columns:
                        label_rows = splitter_df[splitter_df["Name"].notna()]
                        label_rows = label_rows[
                            label_rows["Name"].astype(str).str.strip() != "Document Type Schema Name"
                        ]
                        if "Model Enabled" in label_rows.columns:
                            label_rows = label_rows[
                                ~label_rows["Model Enabled"].astype(str).str.strip().str.lower().isin(["no", "false"])
                            ]
                        item["processor_labels"] = [
                            str(val).strip() for val in label_rows["Name"].tolist() if str(val).strip()
                        ]
                except Exception:
                    pass
        with open(f"{extractors_dir}/Splitter Processors.json", "w", encoding="utf-8") as json_file:
            json.dump(splitter_records, json_file, indent=4, default=str)
    
    for processor in Extractors["Processor Name"]:
        if processor in processor_name_replacements.values():
            key = next((k for k, v in processor_name_replacements.items() if v == processor), None)
            processor_df = pd.read_excel(excel_file_path, sheet_name=key)
        else:
            processor_df = pd.read_excel(excel_file_path, sheet_name=processor)
        processor_df = processor_df.astype({'Page Order Sort': 'float64'})
        write_json(Dataframe =  processor_df, Filepath = f"{extractors_dir}/{processor}.json", key_replacements=key_replacements)
        try: processor_df.close()
        except:pass
    try:
        Obsolete_df = pd.read_excel(excel_file_path, sheet_name="ObsoleteFields")
        write_json(Dataframe =  Obsolete_df, Filepath = f"{extractors_dir}/01Obsolete Fields.json", key_replacements=key_replacements)
    except:
        pass
    try: excel_file.close()
    except:pass
    try: Trained_Models_df.close()
    except:pass
    try: Obsolete_df.close()
    except:pass
        
if __name__ == '__main__':
    for excel_file in [filename for dirpath, _, filenames in os.walk("./") for filename in filenames if filename.lower().endswith(".xlsx")]:
        Excel_to_Json(excel_file_path = excel_file, Test_Sheet_Names = False, Organization = None)
    # Excel_to_Json(excel_file_path = "ISGS Well Completion Schema.xlsx", Test_Sheet_Names = False, Organization = None)
    # Excel_to_Json(excel_file_path = "CALGEM Well Summary Schema.xlsx", Test_Sheet_Names = False, Organization = None)
    # Excel_to_Json(excel_file_path = "Osage Nation Schema.xlsx", Test_Sheet_Names = False, Organization = None)
    # Excel_to_Json(excel_file_path = "NEWTS Schema.xlsx", Test_Sheet_Names = False, Organization = None)
    # """OSAGE NATION SCHEMA: 
    #     Following running this script two occurences of 'Osage_V1_Final_Report_Compl_Dep' should be edited to be 
    #     'Osage_V1_Final_Report_Compl_Deep'. 
    #     This is caused by excel's sheet name character limit that is not present in Google DocAI or Google Sheets.
    #     Occurences: 
    #         Inside Extractor Processors.json as "Processor Name": "Osage_V1_Final_Report_Compl_Deep", and
    #         File name Osage_V1_Final_Report_Compl_Dep.json
    # """
    

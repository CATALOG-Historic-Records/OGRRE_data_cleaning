"""
Created on Tue Mar 18 2025

@author: MichaelPesce
"""
import importlib.resources
import json

def get_processor_list(collaborator: str):
    """Get list of processors for a given collaborator.

        Args:
            collaborator: eg. isgs

        Returns:
            List of processors or None
    """
    try:
        with importlib.resources.open_text(__package__+f".{collaborator}_extractors", "00Extractor Processors.json") as f:
            return json.load(f)
    except Exception as e:
        print(f"unable to find processor list for collaborator: {collaborator}")
        return None
    
def get_processor_by_id(collaborator: str, processor_id: str):
    """Get processor data for given processor id.

        Args:
            collaborator: str = isgs
            processor_id: str

        Returns:
            Dict containing processor data, attributes or None
    """
    if not processor_id:
        return None

    extractor_data = get_processor_list(collaborator)

    processor_data = None
    for extractor in extractor_data:
        if extractor.get("Processor ID") == processor_id:
            processor_data = extractor
            break
    if not processor_data:
        print(f"unable to find processor data for {collaborator} id: {processor_id}")
        return None
    
    processor_name = processor_data.get("Processor Name")

    try:
        with importlib.resources.open_text(__package__+f".{collaborator}_extractors", f"{processor_name}.json") as f:
            processor_data["attributes"] = json.load(f)
    except Exception as e:
        print(f"unable to find attributes for {collaborator} id: {processor_id}")

    return processor_data

def get_processor_by_name(collaborator: str = "isgs", processor_name: str = None):
    """Get processor data for given processor name.

        Args:
            collaborator: str = isgs
            processor_name: str

        Returns:
            Dict containing processor data, attributes or None
    """
    if not processor_name:
        return None

    extractor_data = get_processor_list(collaborator)

    processor_data = None
    for extractor in extractor_data:
        if extractor.get("Processor Name") == processor_name:
            processor_data = extractor
            break
    if not processor_data:
        print(f"unable to find processor data for {collaborator} named: {processor_name}")
        return None
    
    try:
        with importlib.resources.open_text(__package__+f".{collaborator}_extractors", f"{processor_name}.json") as f:
            processor_data["attributes"] = json.load(f)
    except Exception as e:
        print(f"unable to find attributes for {collaborator} named: {processor_name}")

    return processor_data
    
import importlib.resources
import json

def get_processor_list(collaborator: str):
    try:
        with importlib.resources.open_text(__package__+f".{collaborator}_extractors", "00Extractor Processors.json") as f:
            return json.load(f)
    except Exception as e:
        print(f"unable to find processor list for collaborator: {collaborator}")
        return None
    
def get_processor_by_id(collaborator: str, processor_id: str):
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

def get_processor_image(collaborator: str, processor_name: str):
    if processor_name is None:
        return None
    try:
        with importlib.resources.open_text(__package__+f".{collaborator}_extractors.images", f"{processor_name}.png") as f:
            return f.read()
    except Exception as e:
        print(f"unable to find processor images for {collaborator} : {processor_name}")
        return None
    
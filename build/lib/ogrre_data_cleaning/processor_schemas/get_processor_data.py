import importlib.resources
import json

def get_processor_list(collaborator: str = "isgs"):
    with importlib.resources.open_text(__package__+f".{collaborator}_extractors", "00Extractor Processors.json") as f:
        return json.load(f)
    
def get_processor_by_id(collaborator: str = "isgs", processor_id: str = None):
    if not processor_id:
        return None

    with importlib.resources.open_text(__package__+f".{collaborator}_extractors", "00Extractor Processors.json") as f:
        extractor_data = json.load(f)

    processor_data = None
    for extractor in extractor_data:
        if extractor.get("Processor ID") == processor_id:
            processor_data = extractor
            break
    if not processor_data:
        return None
    
    processor_name = processor_data.get("name")

    with importlib.resources.open_text(__package__+f".{collaborator}_extractors", f"{processor_name}.json") as f:
        processor_data["attributes"] = json.load(f)

    return processor_data

def get_processor_by_name(collaborator: str = "isgs", processor_name: str = None):
    if not processor_name:
        return None

    with importlib.resources.open_text(__package__+f".{collaborator}_extractors", "00Extractor Processors.json") as f:
        extractor_data = json.load(f)

    processor_data = None
    for extractor in extractor_data:
        if extractor.get("Processor Name") == processor_name:
            processor_data = extractor
            break
    if not processor_data:
        return None
    
    with importlib.resources.open_text(__package__+f".{collaborator}_extractors", f"{processor_name}.json") as f:
        processor_data["attributes"] = json.load(f)

    return processor_data

def get_processor_image(collaborator: str = "isgs", processor_name: str = None):
    with importlib.resources.open_text(__package__+f".{collaborator}_extractors.images", f"{processor_name}.png") as f:
        return f.read()
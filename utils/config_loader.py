import yaml

#/Users/rajankumar/document_portal/config
def load_config(config_path: str = "/Users/rajankumar/document_portal/config/config.yaml") -> dict:
     with open(config_path, "r") as file:
         config=yaml.safe_load(file)
     return config
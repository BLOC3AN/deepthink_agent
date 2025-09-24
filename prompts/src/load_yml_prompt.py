import yaml

def load_yml_prompt(file_path):
    with open(file_path, 'r') as file:
        prompt = yaml.safe_load(file)
    return prompt

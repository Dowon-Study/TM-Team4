import os
import json

def read_json_file(file_path):
    """
    Reads a JSON file and returns the content as a dictionary.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def write_json_file(data, file_path):
    """
    Writes a dictionary to a JSON file.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def read_text_file(file_path):
    """
    Reads a text file and returns the content as a string.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def write_text_file(content, file_path):
    """
    Writes a string to a text file.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def list_files_in_directory(directory_path):
    """
    Returns a list of files in the specified directory.
    """
    return [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
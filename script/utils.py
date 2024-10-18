
import os
import jsonlines
import json
import xml.etree.ElementTree as ET
import pickle as pkl

def remove_noise(text):
    answer_start = text.find(':') + 1 if ':' in text else 0
    answer_start = text.find('is') + 2 if 'is' in text else answer_start
    text = text[answer_start:].strip().replace('.', '').replace('"', '')
    return text

def load_jsonl(path):
    fi = jsonlines.open(path, 'r')
    text = [line for line in fi]
    return text

def save_pkl(data, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as file:
        pkl.dump(data, file)

def load_pkl(filename):
    with open(filename, "rb") as file:
        data = pkl.load(file)
    return data

def save_json(data, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def reverse_cause(askfor):
    if askfor == 'cause':
        return 'effect'
    else:
        return 'cause'

def parse_copa_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    result_list = []
    for item in root.findall('item'):
        item_id = item.get('id')
        asks_for = item.get('asks-for')
        most_plausible_alternative = item.get('most-plausible-alternative')
        effect = item.find('p').text
        if most_plausible_alternative == '1':
            cause = item.find('a1').text
        else:
            cause = item.find('a2').text
        item_dict = {
            'id': item_id,
            'cause': cause,
            'effect': effect
        }
        result_list.append(item_dict)

    return result_list

def load_from_txt(file_name):
    with open(file_name, "r") as f:
        data = [line.strip() for line in f]
    return data


def save_to_txt(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        f.write(data)

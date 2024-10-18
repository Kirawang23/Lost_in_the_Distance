
from script.utils import *
from datasets import load_dataset
import requests
import gzip
import pandas as pd
import io
import os
import itertools
import nltk
import random
nltk.download('wordnet')
from nltk.corpus import wordnet as wn

def get_data(args):
    shuffler = random.Random(args.seed)
    if args.task_name == "revcause":
        if os.path.exists(os.path.join(args.data_path, 'test.json')):
            dt = load_json(os.path.join(args.data_path, 'test.json'))
        else:
            dt_ = load_jsonl(os.path.join(args.data_path, 'dev.jsonl'))
            shuffler.shuffle(dt_)
            dt = []
            for i in range(0, len(dt_) - 1, 2):
                item1 = dt_[i]
                item2 = dt_[i + 1]
                ask_for_1 = item1['ask-for'].strip().lower()
                ask_for_2 = item2['ask-for'].strip().lower()
                question_1 = '"'+item1['premise'].lower().strip().replace(',', '').replace('.','').replace('"', '')+'"'
                question_2 = '"'+item2['premise'].lower().strip().replace(',', '').replace('.','').replace('"', '')+'"'
                answer_1 = '"'+item1[f"hypothesis{item1['label'] + 1}"].lower().strip().replace(',', '').replace('.','').replace('"', '')+'"'
                answer_2 = '"'+item2[f"hypothesis{item2['label'] + 1}"].lower().strip().replace(',', '').replace('.','').replace('"', '')+'"'
                if question_1 != question_2 and answer_1 != answer_2:
                    group = {
                        f'{reverse_cause(ask_for_1)}1': question_1,
                        f'{ask_for_1}1': answer_1,
                        f'{reverse_cause(ask_for_2)}2': question_2,
                        f'{ask_for_2}2': answer_2
                    }
                    dt.append(group)

            if len(dt) < args.data_size:
                raise ValueError("Not enough data to select from.")
            save_json(dt, os.path.join(args.data_path, 'test.json'))

    elif args.task_name == "revcause-copa":
        dt = parse_copa_xml(args.data_path + 'copa-all.xml')

    elif args.task_name == 'revname':
        if os.path.exists(os.path.join(args.data_path, 'test.json')):
            dt = load_json(os.path.join(args.data_path, 'test.json'))
        else:
            names = load_from_txt(os.path.join(args.data_path, 'names.txt'))
            names = [n.lower().strip().replace(',', '').replace('.','').replace('"', '') for n in names]
            names = [' '.join([m.capitalize() for m in n.split(" ")]) for n in names]
            descriptions = load_from_txt(os.path.join(args.data_path, 'descriptions.txt'))
            descriptions = [d.lower().strip().replace(',', '').replace('.','').replace('"', '') for d in descriptions]
            all_combinations = list(itertools.product(names, descriptions))
            shuffler.shuffle(all_combinations)
            dt = []
            for i in range(0, len(all_combinations) - 1, 2):
                name1, description1 = all_combinations[i]
                name2, description2 = all_combinations[i + 1]
                if name1 != name2 and description1 != description2:
                    group = {
                        'name1': name1,
                        'description1': description1,
                        'name2': name2,
                        'description2': description2
                    }
                    dt.append(group)

            if len(dt) < args.data_size:
                raise ValueError("Not enough data to select from.")
            save_json(dt, os.path.join(args.data_path, 'test.json'))

    elif args.task_name == 'revparent':

        if os.path.exists(os.path.join(args.data_path, 'test.json')):
            dt = load_json(os.path.join(args.data_path, 'test.json'))
        else:
            dt_ = pd.read_csv(args.data_path+'parent_child_pairs.csv').to_dict('records')
            shuffler.shuffle(dt_)
            dt = []
            for i in range(0, len(dt_) - 1, 2):
                item1 = dt_[i]
                item2 = dt_[i + 1]

                type_1 = item1['parent_type'].lower().strip().replace(',', '').replace('.', '').replace('"', '')
                type_2 = item2['parent_type'].lower().strip().replace(',', '').replace('.', '').replace('"', '')

                parent1 = item1['parent'].lower().strip().replace(',', '').replace('.', '').replace('"', '')
                parent1 = ' '.join([m.capitalize() for m in parent1.split(" ")])

                child1 = item1['child'].lower().strip().replace(',', '').replace('.', '').replace('"', '')
                child1 = ' '.join([m.capitalize() for m in child1.split(" ")])

                parent2 = item2['parent'].lower().strip().replace(',', '').replace('.', '').replace('"', '')
                parent2 = ' '.join([m.capitalize() for m in parent2.split(" ")])

                child2 = item2['child'].lower().strip().replace(',', '').replace('.', '').replace('"', '')
                child2 = ' '.join([m.capitalize() for m in child2.split(" ")])

                if parent1 != parent2 and child1 != child2:
                    group = {
                        f'parent1': parent1,
                        f'child1': child1,
                        f'type1': type_1,
                        f'parent2': parent2,
                        f'child2': child2,
                        f'type2': type_2
                    }
                    dt.append(group)

            if len(dt) < args.data_size:
                raise ValueError("Not enough data to select from.")
            save_json(dt, os.path.join(args.data_path, 'test.json'))


    shuffler.shuffle(dt)
    return dt

def get_noise(args):

    if args.noise_type == 'random':
        num_word = 1000000
        words = [n for n in wn.all_lemma_names() if "_" not in n]
        random.shuffle(words)
        noise = " ".join(words[:num_word])
        return noise

    elif args.noise_type == 'novel':
        all_files = [f for f in os.listdir(args.noise_path) if f.endswith('.txt')]
        noise = load_from_txt(args.noise_path + random.choice(all_files))
        noise = ' '.join(noise).replace('\n', '')
        print("*"*30, f"Loaded {args.noise_path + random.choice(all_files)}", "*"*30)
        return noise

    elif args.noise_type == 'fineweb':
        if os.path.exists(args.noise_path + 'fineweb.txt'):
            noise = load_from_txt(args.noise_path + 'fineweb.txt')
            noise = ' '.join(noise).replace('\n', '')[: 1000000]
        else:
            noise = load_dataset("HuggingFaceFW/fineweb", data_files="sample/10BT/001_00000.parquet")
            noise = noise['train']['text']
            noise = ' '.join(noise).replace('\n','')[: 1000000]
            save_to_txt(noise, args.noise_path + 'fineweb.txt')
        return noise

    elif args.noise_type == 'redpajama':
        if os.path.exists(args.noise_path + 'redpajama.txt'):
            noise = load_from_txt(args.noise_path + 'redpajama.txt')
            noise = ' '.join(noise).replace('\n', '')[: 1000000]
        else:
            url = "https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2/resolve/main/sample/documents/2023-06/0000/en_head.json.gz"
            response = requests.get(url)
            with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as gz:
                noise = []
                for line in gz:
                    noise.append(json.loads(line.decode('utf-8'))['raw_content'])
            noise = ' '.join(noise).replace('\n', '')[: 1000000]
            save_to_txt(noise, args.noise_path + 'redpajama.txt')
        return noise

def split_noise(noise, max_tokens=3000):
    noise = noise.replace('\n', ' ').strip()
    noise_tokens = noise.split() if ' ' in noise else noise
    chunk_size = max_tokens // 3
    noise_length = len(noise_tokens)

    if noise_length < max_tokens:
        raise ValueError("Not enough noise tokens to select from.")

    selected_noise_tokens = random.sample(noise_tokens, max_tokens)
    noise_chunk1 = " ".join(selected_noise_tokens[:chunk_size])
    noise_chunk2 = " ".join(selected_noise_tokens[chunk_size:2 * chunk_size])
    noise_chunk3 = " ".join(selected_noise_tokens[2 * chunk_size: 3 * chunk_size])

    if len(noise_chunk1.split()) != chunk_size or len(noise_chunk2.split()) != chunk_size or len(noise_chunk3.split()) != chunk_size:
        raise ValueError("Noise chunks are not of equal length.")

    return noise_chunk1, noise_chunk2, noise_chunk3

def generate_combined_strings(first, second, noise, max_tokens=3000, task_name='revcause',n_shot=None, select_qa=None):
    noise_chunk1, noise_chunk2, noise_chunk3 = split_noise(noise, max_tokens)

    if select_qa:
        if task_name == 'revname':
            args_start_name_1 = random.choice([1, 2])
            args_start_name_2 = 2 if args_start_name_1 == 1 else 1
            args_start_description_1 = random.choice([1, 2])
            args_start_description_2 = 2 if args_start_description_1 == 1 else 1
            name1 = first[f'name{args_start_name_1}']
            name1 = ' '.join([m.capitalize() for m in name1.split(' ')])
            name2 = first[f'name{args_start_name_2}']
            name2 = ' '.join([m.capitalize() for m in name2.split(' ')])
            description1 = first[f'description{args_start_description_1}']
            description2 = first[f'description{args_start_description_2}']
        elif task_name == 'revcause':
            args_start_cause_1 = random.choice([1, 2])
            args_start_cause_2 = 2 if args_start_cause_1 == 1 else 1
            args_start_effect_1 = random.choice([1, 2])
            args_start_effect_2 = 2 if args_start_effect_1 == 1 else 1
            cause1 = first[f'cause{args_start_cause_1}']
            effect1 = first[f'effect{args_start_effect_1}']
            cause2 = first[f'cause{args_start_cause_2}']
            effect2 = first[f'effect{args_start_effect_2}']
        elif task_name == 'revparent':
            args_start_parent_1 = random.choice([1, 2])
            args_start_parent_2 = 2 if args_start_parent_1 == 1 else 1
            args_start_child_1 = random.choice([1, 2])
            args_start_child_2 = 2 if args_start_child_1 == 1 else 1
            parent1 = first[f'parent{args_start_parent_1}']
            child1 = first[f'child{args_start_child_1}']
            type1 = first[f'type{args_start_child_1}']
            parent2 = first[f'parent{args_start_parent_2}']
            child2 = first[f'child{args_start_child_2}']
            type2 = first[f'type{args_start_child_2}']


    if 'revname' in task_name or 'revdesc' in task_name:
            args_start_name_1 = 'first' if args_start_name_1 == 1 else 'second'
            args_start_name_2 = 'first' if args_start_name_2 == 1 else 'second'
            args_start_description_1 = 'first' if args_start_description_1 == 1 else 'second'
            args_start_description_2 = 'first' if args_start_description_2 == 1 else 'second'
            start_name1 = f'The {args_start_name_1} person is'
            start_description1 = f'The {args_start_description_1} person is'
            start_name2 = f'The {args_start_name_2} person is'
            start_description2 = f'The {args_start_description_2} person is'
    elif 'revcause' in task_name:
            args_start_cause_1 = 'first' if args_start_cause_1 == 1 else 'second'
            args_start_cause_2 = 'first' if args_start_cause_2 == 1 else 'second'
            args_start_effect_1 = 'first' if args_start_effect_1 == 1 else 'second'
            args_start_effect_2 = 'first' if args_start_effect_2 == 1 else 'second'
            start_cause1 = f'The {args_start_cause_1} event is'
            start_effect1 = f"The {args_start_effect_1} event\'s effect is"
            start_cause2 = f'The {args_start_cause_2} event is'
            start_effect2 = f"The {args_start_effect_2} event\'s effect is"
    elif 'revparent' in task_name:
            args_start_parent_1 = 'first' if args_start_parent_1 == 1 else 'second'
            args_start_parent_2 = 'first' if args_start_parent_2 == 1 else 'second'
            args_start_child_1 = 'first' if args_start_child_1 == 1 else 'second'
            args_start_child_2 = 'first' if args_start_child_2 == 1 else 'second'
            start_parent1 = f'The {args_start_parent_1} person is'
            start_child1 = f'The {args_start_child_1} person\'s child is'
            start_parent2 = f'The {args_start_parent_2} person is'
            start_child2 = f'The {args_start_child_2} person\'s child is'


    if task_name == 'revname':
        combinations = {
            'qa': f"{start_name1} {name1}. {start_name2} {name2}. {start_description1} {description1}. {start_description2} {description2}.",
            'qna': f"{start_name1} {name1}. {start_name2} {name2}. {noise_chunk1}. {start_description1} {description1}. {start_description2} {description2}.",
            'qnna': f"{start_name1} {name1}. {start_name2} {name2}. {noise_chunk1} {noise_chunk2}. {start_description1} {description1}. {start_description2} {description2}.",
            'qnnna': f"{start_name1} {name1}. {start_name2} {name2}. {noise_chunk1} {noise_chunk2} {noise_chunk3}. {start_description1} {description1}. {start_description2} {description2}.",
            'qannn': f"{start_name1} {name1}. {start_name2} {name2}. {start_description1} {description1}. {start_description2} {description2}. {noise_chunk1} {noise_chunk2} {noise_chunk3}.",
            'nnnqa': f"{noise_chunk1} {noise_chunk2} {noise_chunk3}. {start_name1} {name1}. {start_name2} {name2}. {start_description1} {description1}. {start_description2} {description2}.",
            'nqann': f"{noise_chunk1}. {start_name1} {name1}. {start_name2} {name2}. {start_description1} {description1}. {start_description2} {description2}. {noise_chunk2} {noise_chunk3}.",
            'nnqan': f"{noise_chunk1} {noise_chunk2}. {start_name1} {name1}. {start_name2} {name2}. {start_description1} {description1}. {start_description2} {description2}. {noise_chunk3}.",

        }
    elif task_name == 'revparent':
        combinations = {
            'qa': f"{start_parent1} {parent1}. {start_parent2} {parent2}. {start_child1} {child1}. {start_child2} {child2}.",
            'qna': f"{start_parent1} {parent1}. {start_parent2} {parent2}. {noise_chunk1} {start_child1} {child1}. {start_child2} {child2}.",
            'qnna': f"{start_parent1} {parent1}. {start_parent2} {parent2}. {noise_chunk1} {noise_chunk2} {start_child1} {child1}. {start_child2} {child2}.",
            'qnnna': f"{start_parent1} {parent1}. {start_parent2} {parent2}. {noise_chunk1} {noise_chunk2} {noise_chunk3} {start_child1} {child1}. {start_child2} {child2}.",
        }
    elif task_name == 'revcause':
        combinations = {
            'qa': f"{start_cause1} {cause1}. {start_cause2} {cause2}. {start_effect1} {effect1}. {start_effect2} {effect2}.",
            'qna': f"{start_cause1} {cause1}. {start_cause2} {cause2}. {noise_chunk1} {start_effect1} {effect1}. {start_effect2} {effect2}.",
            'qnna': f"{start_cause1} {cause1}. {start_cause2} {cause2}. {noise_chunk1} {noise_chunk2} {start_effect1} {effect1}. {start_effect2} {effect2}.",
            'qnnna': f"{start_cause1} {cause1}. {start_cause2} {cause2}. {noise_chunk1} {noise_chunk2} {noise_chunk3} {start_effect1} {effect1}. {start_effect2} {effect2}.",
        }
    return combinations

def add_noise(entry, noise, args):
    if 'cause' in args.task_name:
        if args.n_shot > 0:
            args.select_qa = random.randint(1, args.n_shot)
        else:
            args.select_qa = random.randint(1, 2)
        entry['no-noise'] = \
        generate_combined_strings(entry, entry, noise, args.max_tokens, args.task_name, args.n_shot, args.select_qa)['qa']
        entry['context'] = \
        generate_combined_strings(entry, entry, noise, args.max_tokens, args.task_name, args.n_shot, args.select_qa)[args.position]
        if args.reverse:
            entry['question'] = f"What is the cause of {entry[f'effect{args.select_qa}']}?"
            entry['answer'] = entry[f'cause{args.select_qa}']
        else:
            entry['question'] = f"What is effect of {entry[f'cause{args.select_qa}']}?"
            entry['answer'] = entry[f'effect{args.select_qa}']

    elif args.task_name == 'revname':
        if args.n_shot > 0:
            args.select_qa = random.randint(1, args.n_shot)
        else:
            args.select_qa = random.randint(1, 2)
        entry['no-noise'] = \
        generate_combined_strings(entry, entry, noise, args.max_tokens, args.task_name, args.n_shot, args.select_qa)['qa']
        entry['context'] = \
        generate_combined_strings(entry, entry, noise, args.max_tokens, args.task_name, args.n_shot, args.select_qa)[args.position]
        if args.reverse:
            entry['question'] = f"What is the name of {entry[f'description{args.select_qa}']}?"
            entry['answer'] = entry[f'name{args.select_qa}']
        else:
            entry['question'] = f"What is {entry[f'name{args.select_qa}']} known as?"
            entry['answer'] = entry[f'description{args.select_qa}']

    elif args.task_name == 'revparent':
        if args.n_shot > 0:
            args.select_qa = random.randint(1, args.n_shot)
        else:
            args.select_qa = random.randint(1, 2)
        entry['no-noise'] = \
        generate_combined_strings(entry, entry, noise, args.max_tokens, args.task_name, args.n_shot, args.select_qa)['qa']
        entry['context'] = \
        generate_combined_strings(entry, entry, noise, args.max_tokens, args.task_name, args.n_shot, args.select_qa)[args.position]
        if args.reverse:
            entry['question'] = f"What is the name of {entry[f'child{args.select_qa}']}\'s parent?"
            entry['answer'] = entry[f'parent{args.select_qa}']
        else:
            entry['question'] = f"What is name of {entry[f'parent{args.select_qa}']}\'s child?"
            entry['answer'] = entry[f'child{args.select_qa}']
    return entry




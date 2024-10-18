
import re
import matplotlib.pyplot as plt
import pandas as pd
import os

def parse_logs(log_folder):
    task_name_pattern = re.compile(r"【task】: (.+?) 【model】")
    model_pattern = re.compile(r"【model】: (.+?) 【noise_type】")
    noise_type_pattern = re.compile(r"【noise_type】: (.+?) 【n_shot】")
    n_shot_pattern = re.compile(r"【n_shot】: (.+?) 【reverse】")
    reverse_pattern = re.compile(r"【reverse】: (.+?) 【position】")
    position_pattern = re.compile(r"【position】: (.+?) 【max_tokens】")
    max_token_pattern = re.compile(r"【max_tokens】: (.+?)\s")
    exact_match_pattern = re.compile(r"Exact Match: ([\d.]+)")
    precision_pattern = re.compile(r"Precison: ([\d.]+)")
    recall_pattern = re.compile(r"Recall: ([\d.]+)")
    f1_score_pattern = re.compile(r"F1 Score: ([\d.]+)")

    data = []

    for filename in os.listdir(log_folder):
        if not filename.endswith(".log") and not filename.startswith("."):
            subfolder = os.path.join(log_folder, filename)
            for subfilename in os.listdir(subfolder):
                if subfilename.endswith(".log"):
                    with open(os.path.join(subfolder, subfilename), 'r') as file:
                        content = file.read()

                        model = model_pattern.search(content)
                        noise_type = noise_type_pattern.search(content)
                        n_shot = n_shot_pattern.search(content)
                        task_name = task_name_pattern.search(content)
                        reverse = reverse_pattern.search(content)
                        position = position_pattern.search(content)
                        if not position:
                            position_pattern = re.compile(r"【position】: (.+?)\s")
                            position = position_pattern.search(content)
                        exact_match = exact_match_pattern.search(content)
                        f1_score = f1_score_pattern.search(content)
                        precision = precision_pattern.search(content)
                        recall = recall_pattern.search(content)
                        max_token = max_token_pattern.search(content)

                        if model and noise_type and n_shot and task_name and reverse and position and exact_match and f1_score and precision and recall and max_token:
                            model_name = model.group(1)
                            if '2024' in model_name:
                                model_name = '-'.join(model_name.split("-")[:-1])

                            data.append({
                                'model': model_name,
                                'noise_type': noise_type.group(1),
                                'n-shot': n_shot.group(1),
                                'task_name': task_name.group(1),
                                'reverse': reverse.group(1),
                                'position': position.group(1),
                                'ExactMatch': round(float(exact_match.group(1)) * 100, 2),
                                'F1': round(float(f1_score.group(1)) * 100, 2),
                                'Precision': round(float(precision.group(1)) * 100, 2),
                                'Recall': round(float(recall.group(1)) * 100, 2),
                                'max_token': int(max_token.group(1))
                            })
                        elif model and noise_type and n_shot and task_name and reverse and position and exact_match and f1_score and precision and recall:
                            model_name = model.group(1)
                            if '2024' in model_name:
                                model_name = '-'.join(model_name.split("-")[:-1])

                            data.append({
                                'model': model_name,
                                'noise_type': noise_type.group(1),
                                'n-shot': n_shot.group(1),
                                'task_name': task_name.group(1),
                                'reverse': reverse.group(1),
                                'position': position.group(1),
                                'ExactMatch': round(float(exact_match.group(1)) * 100, 2),
                                'F1': round(float(f1_score.group(1)) * 100, 2),
                                'Precision': round(float(precision.group(1)) * 100, 2),
                                'Recall': round(float(recall.group(1)) * 100, 2)
                            })

        elif filename.endswith(".log"):
            with open(os.path.join(log_folder, filename), 'r') as file:
                content = file.read()

                model = model_pattern.search(content)
                noise_type = noise_type_pattern.search(content)
                n_shot = n_shot_pattern.search(content)
                task_name = task_name_pattern.search(content)
                reverse = reverse_pattern.search(content)
                position = position_pattern.search(content)
                if not position:
                    position_pattern = re.compile(r"【position】: (.+?)\s")
                    position = position_pattern.search(content)
                exact_match = exact_match_pattern.search(content)
                f1_score = f1_score_pattern.search(content)
                precision = precision_pattern.search(content)
                recall = recall_pattern.search(content)
                max_token = max_token_pattern.search(content)

                if model and noise_type and n_shot and task_name and reverse and position and exact_match and f1_score and precision and recall and max_token:
                    model_name = model.group(1)
                    if '2024' in model_name:
                        model_name = '-'.join(model_name.split("-")[:-1])
                    data.append({
                        'model': model_name,
                        'noise_type': noise_type.group(1),
                        'n-shot': n_shot.group(1),
                        'task_name': task_name.group(1),
                        'reverse': reverse.group(1),
                        'position': position.group(1),
                        'ExactMatch': round(float(exact_match.group(1)) * 100, 2),
                        'F1': round(float(f1_score.group(1)) * 100, 2),
                        'Precision': round(float(precision.group(1)) * 100, 2),
                        'Recall': round(float(recall.group(1)) * 100, 2),
                        'max_token': int(max_token.group(1))
                    })
                elif model and noise_type and n_shot and task_name and reverse and position and exact_match and f1_score and precision and recall:
                    model_name = model.group(1)
                    if '2024' in model_name:
                        model_name = '-'.join(model_name.split("-")[:-1])
                    data.append({
                        'model': model_name,
                        'noise_type': noise_type.group(1),
                        'n-shot': n_shot.group(1),
                        'task_name': task_name.group(1),
                        'reverse': reverse.group(1),
                        'position': position.group(1),
                        'ExactMatch': round(float(exact_match.group(1)) * 100, 2),
                        'F1': round(float(f1_score.group(1)) * 100, 2),
                        'Precision': round(float(precision.group(1)) * 100, 2),
                        'Recall': round(float(recall.group(1)) * 100, 2)
                    })

    return pd.DataFrame(data)

def lost_in_the_distance_different_noise_length(df, save_path):
    colors = ['#1f77b4', '#ff7f0e']

    df = df[
        (df['task_name'] == 'revname') &
        (df['noise_type'] == 'novel') &
        (df['n-shot'] == '0') &
        (df['model'] == 'claude-3-5-sonnet') &
        (df['max_token']>10000)
        ]

    position_order = ["qa", "qna", "qnna", "qnnna"]
    position_order_name = ["AB", "ANB", "ANNB", "ANNNB"]

    df = df[df['position'].isin(position_order)]
    df = df.dropna()

    token_groups = df.groupby(['max_token', 'reverse'])
    plt.figure(figsize=(5,4))

    for idx, (token_group, group) in enumerate(token_groups):
        max_token, reverse = token_group
        max_token = int(max_token)
        group['position'] = pd.Categorical(group['position'], categories=position_order, ordered=True)
        group = group.sort_values('position')
        label = f"claude-3.5-sonnet (Backward)" if reverse == 'True' else f"claude-3.5-sonnet (Forward)"
        linestyle = '--' if reverse == 'True' else '-'
        marker = 'o' if reverse == 'True' else '^'

        x = group['position']
        y = group['F1']
        plt.plot(x, y, label=label, color=colors[idx // 2], linestyle=linestyle, marker=marker)

    plt.title('Lost in the Distance (Noise Length=20,000)', fontsize=fontsize1-2)
    plt.ylabel('F1 Score', fontsize=fontsize2)
    plt.xlabel('Position', fontsize=fontsize2)
    plt.xticks(position_order, position_order_name, fontsize=fontsize2)
    plt.legend(loc='lower left', fontsize=fontsize3)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f'{save_path}/lost_in_the_distance_different_noise_length.png')
    plt.show()

def lost_in_the_distance_split(df, save_path):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    df = df[
        (df['task_name'] == 'revname') &
        (df['noise_type'] == 'novel') &
        (df['n-shot'] == '0') &
        (df['model'].isin(['gemini-1.5-pro', 'gemini-1.5-flash', 'gpt-4', 'gpt-4o-mini']))
    ]

    position_order = ["qa", "qna", "qnna", "qnnna"]
    position_order_name = ["AB", "ANB", "ANNB", "ANNNB"]

    df = df[df['position'].isin(position_order)]
    df = df.dropna()

    model_groups = df.groupby('model')
    fig, axs = plt.subplots(2, 1, figsize=(5, 6))
    plt.subplots_adjust(hspace=0.5)


    for idx, (model, group) in enumerate(model_groups):
        forward_group = group[group['reverse'] == 'False']

        if not forward_group.empty:
            forward_group['position'] = pd.Categorical(forward_group['position'], categories=position_order, ordered=True)
            forward_group = forward_group.sort_values('position')

            x = forward_group['position']
            y = forward_group['F1']
            axs[0].plot(x, y, label=model, color=colors[idx % len(colors)], linestyle='-', marker='^')

    axs[0].set_title('Lost in the Distance (Forward)', fontsize=fontsize1)
    axs[0].set_ylabel('F1 Score', fontsize=fontsize2)
    axs[0].set_xlabel('Position', fontsize=fontsize2)
    axs[0].set_xticks(position_order, position_order_name, fontsize=fontsize2)
    axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2, fontsize=fontsize3)


    for idx, (model, group) in enumerate(model_groups):
        reverse_group = group[group['reverse'] == 'True']

        if not reverse_group.empty:
            reverse_group['position'] = pd.Categorical(reverse_group['position'], categories=position_order, ordered=True)
            reverse_group = reverse_group.sort_values('position')

            x = reverse_group['position']
            y = reverse_group['F1']
            axs[1].plot(x, y, label=model, color=colors[idx % len(colors)], linestyle='--', marker='o')

    axs[1].set_title('Lost in the Distance (Backward)', fontsize=fontsize1)
    axs[1].set_ylabel('F1 Score', fontsize=fontsize2)
    axs[1].set_xlabel('Position', fontsize=fontsize2)
    axs[1].set_xticks(position_order, position_order_name, fontsize=fontsize2)
    axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2, fontsize=fontsize3)

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f'{save_path}/lost_in_the_distance_different_direction.png')
    plt.show()

def no_distance_no_degradation_split(df, save_path):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    df = df[(df['task_name'] == 'revname') & (df['noise_type'] == 'novel') & (df['n-shot'] == '0') &
            (df['model'].isin(['gemini-1.5-pro', 'gemini-1.5-flash', 'gpt-4', 'gpt-4o-mini']))]

    position_order = ["qnnna", "qannn", "nnnqa", "nqann", "nnqan"]
    position_order_name = ["ANNNB", "ABNNN", "NNNAB", "NABNN", "NNABN"]
    df = df[df['position'].isin(position_order)]
    df = df.dropna()
    model_groups = df.groupby('model')
    fig, axs = plt.subplots(2, 1, figsize=(5, 6))
    plt.subplots_adjust(hspace=0.5)

    for idx, (model, group) in enumerate(model_groups):
        forward_group = group[group['reverse'] == 'False']

        if not forward_group.empty:
            forward_group['position'] = pd.Categorical(forward_group['position'], categories=position_order,
                                                       ordered=True)
            forward_group = forward_group.sort_values('position')

            x = forward_group['position']
            y = forward_group['F1']
            axs[0].plot(x, y, label=model, color=colors[idx % len(colors)], linestyle='-', marker='^')

    axs[0].set_title('No Distance, No Degradation (Forward)', fontsize=fontsize1)
    axs[0].set_ylabel('F1 Score', fontsize=fontsize2)
    axs[0].set_xlabel('Position', fontsize=fontsize2)
    axs[0].set_xticks(position_order)
    axs[0].set_xticklabels(position_order_name, fontsize=fontsize2)

    axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2, fontsize=fontsize3)

    for idx, (model, group) in enumerate(model_groups):
        reverse_group = group[group['reverse'] == 'True']

        if not reverse_group.empty:
            reverse_group['position'] = pd.Categorical(reverse_group['position'], categories=position_order,
                                                       ordered=True)
            reverse_group = reverse_group.sort_values('position')

            x = reverse_group['position']
            y = reverse_group['F1']
            axs[1].plot(x, y, label=model, color=colors[idx % len(colors)], linestyle='--', marker='o')

    axs[1].set_title('No Distance, No Degradation (Backward)', fontsize=fontsize1)
    axs[1].set_ylabel('F1 Score', fontsize=fontsize2)
    axs[1].set_xlabel('Position', fontsize=fontsize2)
    axs[1].set_xticks(position_order)
    axs[1].set_xticklabels(position_order_name, fontsize=fontsize2)
    axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2, fontsize=fontsize3)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f'{save_path}/no_distance_no_degradation_different_direction.png', bbox_inches='tight')
    plt.show()

def lost_in_the_distance_different_task_split(df, save_path):
    colors = {'gemini-1.5-pro': '#1f77b4', 'gemini-1.5-flash': '#ff7f0e'}
    linestyles = {'Forward': '-', 'Backward': '--'}
    markers = {'Forward': 'o', 'Backward': '^'}
    df = df[(df['model'].isin(['gemini-1.5-pro', 'gemini-1.5-flash'])) & (df['noise_type'] == 'novel') & (df['n-shot'] == '0')]
    position_order = ["qa", "qna", "qnna", "qnnna"]
    position_order_name = ["AB", "ANB", "ANNB", "ANNNB"]
    task_map = {
        'revcause': 'Cause2Effect',
        'revparent': 'Parent2Child'
    }
    tasks_to_plot = ['revcause', 'revparent']
    df = df[df['task_name'].isin(tasks_to_plot)]
    df = df[df['position'].isin(position_order)]
    df = df.dropna()
    task_groups = df.groupby('task_name')
    fig, axes = plt.subplots(2, 1, figsize=(6,5.5))
    plt.subplots_adjust(hspace=0.4)
    legend_lines = []
    legend_labels = []
    for idx, (task, group) in enumerate(task_groups):
        if task not in task_map:
            continue
        task_label = task_map[task]

        group['position'] = pd.Categorical(group['position'], categories=position_order, ordered=True)
        group = group.sort_values('position')
        ax = axes[idx]
        reverse_groups = group.groupby('reverse')
        for reverse, reverse_group in reverse_groups:
            direction_label = 'Backward' if reverse == 'True' else 'Forward'
            for model_name, model_group in reverse_group.groupby('model'):
                label = f"{model_name} ({direction_label})"
                x = model_group['position']
                y = model_group['F1']
                color = colors[model_name]
                linestyle = linestyles[direction_label]
                marker = markers[direction_label]
                line, = ax.plot(x, y, label=label, color=color, linestyle=linestyle, marker=marker)
                if label not in legend_labels:
                    legend_lines.append(line)
                    legend_labels.append(label)

        ax.set_title(f'Lost in the Distance ({task_label})', fontsize=fontsize1)
        ax.set_ylabel('F1 Score', fontsize=fontsize2)
        ax.set_xlabel('Position', fontsize=fontsize2)
        ax.set_xticks(position_order)
        ax.set_xticklabels(position_order_name, fontsize=fontsize2)
    axes[-1].legend(legend_lines, legend_labels, loc='upper center', bbox_to_anchor=(0.5, -0.6), ncol=1, fontsize=fontsize3)
    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f'{save_path}/lost_in_the_distance_different_task.png', bbox_inches='tight')
    plt.show()

def lost_in_the_distance_different_noise_split(df, save_path):
    colors = {'gemini-1.5-pro': '#1f77b4', 'gemini-1.5-flash': '#ff7f0e'}
    linestyles = {'Forward': '-', 'Backward': '--'}
    markers = {'Forward': 'o', 'Backward': '^'}
    df = df[(df['model'].isin(['gemini-1.5-pro', 'gemini-1.5-flash'])) & (df['task_name'] == 'revname') & (df['n-shot'] == '0')]
    position_order = ["qa", "qna", "qnna", "qnnna"]
    position_order_name = ["AB", "ANB", "ANNB", "ANNNB"]
    noise_map = {
        'redpajama': 'RedPajama',
        'random': 'Random'
    }
    noises_to_plot = ['redpajama', 'random']
    df = df[df['noise_type'].isin(noises_to_plot)]
    df = df[df['position'].isin(position_order)]
    df = df.dropna()
    noise_groups = df.groupby('noise_type')
    fig, axes = plt.subplots(2, 1, figsize=(6,5.5))
    plt.subplots_adjust(hspace=0.4)
    legend_lines = []
    legend_labels = []
    for idx, (noise, group) in enumerate(noise_groups):
        if noise not in noise_map:
            continue

        noise_label = noise_map[noise]
        group['position'] = pd.Categorical(group['position'], categories=position_order, ordered=True)
        group = group.sort_values('position')
        ax = axes[idx]
        reverse_groups = group.groupby('reverse')
        for reverse, reverse_group in reverse_groups:
            direction_label = 'Backward' if reverse == 'True' else 'Forward'

            for model_name, model_group in reverse_group.groupby('model'):
                label = f"{model_name} ({direction_label})"
                x = model_group['position']
                y = model_group['F1']
                color = colors[model_name]
                linestyle = linestyles[direction_label]
                marker = markers[direction_label]
                line, = ax.plot(x, y, label=label, color=color, linestyle=linestyle, marker=marker)
                if label not in legend_labels:
                    legend_lines.append(line)
                    legend_labels.append(label)

        ax.set_title(f'Lost in the Distance ({noise_label})', fontsize=fontsize1)
        ax.set_ylabel('F1 Score', fontsize=fontsize2)
        ax.set_xlabel('Position', fontsize=fontsize2)
        ax.set_xticks(position_order)
        ax.set_xticklabels(position_order_name, fontsize=fontsize2)

    axes[-1].legend(legend_lines, legend_labels, loc='upper center', bbox_to_anchor=(0.5, -0.6), ncol=1, fontsize=fontsize3)
    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f'{save_path}/lost_in_the_distance_different_noise.png', bbox_inches='tight')
    plt.show()

fontsize1 = 16
fontsize2 = 12
fontsize3 = 10
log_folder = f'../log/revname/'
df = parse_logs(log_folder)
output_folder = f'../figure/'
lost_in_the_distance_different_noise_length(df, output_folder)
log_folder = f'../log/'
df = parse_logs(log_folder)
lost_in_the_distance_split(df, output_folder)
no_distance_no_degradation_split(df, output_folder)
lost_in_the_distance_different_task_split(df, output_folder)
lost_in_the_distance_different_noise_split(df, output_folder)

from tqdm import trange
from script.dataloader import *
from script.inference import *
import os
from script.utils import remove_noise

def get_examples(data, i, noise, args):
    if args.n_shot > 0:

        end_idx = -(i + 1) * args.n_shot
        start_idx = end_idx + args.n_shot
        start_idx = None if start_idx == 0 else start_idx
        if start_idx is not None:
            for k in range(end_idx, start_idx):
                data[k] = add_noise(data[k], noise, args)
        else:
            for k in range(end_idx, len(data)):
                data[k] = add_noise(data[k], noise, args)
    else:
        example = None
    return example

def get_testdata(args):
    if os.path.exists("predictions/" + args.task_name + "/" + args.save_path + ".pkl"):
        data = load_pkl("predictions/" + args.task_name + "/" + args.save_path + ".pkl")
    else:
        data = get_data(args)
    return data

def run(args, log):
    log.logger.info(f"run_{args.save_path}")
    data = get_testdata(args)
    model = LLM(model=args.model, task_name=args.task_name)

    for i in trange(args.data_size, desc="[Inference:] get LLM predictions"):
        if "prediction" in data[i].keys():
            if data[i]["prediction"] != "":
                print(f"{i} already has predictions")
                continue

        noise = get_noise(args)
        data[i] = add_noise(data[i], noise, args)
        examples = get_examples(data, i, noise, args)

        prediction,prompt_text = model.get_results(data[i], examples, args)
        data[i]["prediction"] = remove_noise(prediction)
        data[i]["answer"] = remove_noise(data[i]["answer"])
        data[i]["prompt"] = prompt_text
        print("*" * 50)
        print(f"prediction: {data[i]['prediction']}")
        print(f"answer: {data[i]['answer']}")

        save_pkl(data,"predictions/" + args.task_name + "/" + args.save_path + ".pkl")
        print(
            f"save predictions to predictions/" + args.task_name + "/" + args.save_path + ".pkl"
        )



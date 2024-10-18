
import argparse
import logging
from script.run import *
import os
from script.evaluate import get_eval_report

class Logger(object):
    def __init__(self, filename, level="info"):
        level = logging.INFO if level == "info" else logging.DEBUG
        self.logger = logging.getLogger(filename)
        self.logger.propagate = False
        self.logger.setLevel(level)
        th = logging.FileHandler(filename, "w")
        self.logger.addHandler(th)

def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('true'):
        return True
    elif value.lower() in ('false'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", default="data/", type=str, help="Data directory")
    parser.add_argument(
        "--data_size", default=200, type=int, help="Data size")
    parser.add_argument(
        "--reverse", default=True, type=str2bool, help="Reverse the task")
    parser.add_argument(
        "--max_tokens", default=10000, type=int, help="Max noise tokens")
    parser.add_argument(
        "--task_name", default="revparent", type=str, help="task to test: revname/revcause/revparent/revcause-copa"
    )
    parser.add_argument(
        "--position", default="qa", type=str, help="qa/qna/qnna/qnnna"
    )
    parser.add_argument(
        "--noise_type", default="novel", type=str, help="random/novel/fineweb/redpajama")
    parser.add_argument(
        "--n_shot", default=0, type=int, help="0-shot/1-shot/2-shot")
    parser.add_argument(
        "--model", default="gemini-1.5-pro", type=str, help="The model to use: gemini-1.5-pro/claude-3-5-sonnet-20240620/gpt-4o/gpt-4o-mini/claude-3-haiku-20240307.\nThe model version: claude-3-5-sonnet-20240620, gpt-4o-2024-08-06, gemini-1.5-pro-002, gpt-4o-mini-2024-07-18, claude-3-haiku-20240307."
    )
    parser.add_argument(
        "--seed", default=42, type=int, help="Random seed.")
    parser.add_argument(
        "--log_dir", default="log/", type=str, help="Path for Logging file"
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_argument()
    args.data_path = args.data_dir + args.task_name + "/"
    args.noise_path = args.data_dir + "noise/" + args.noise_type + "/"
    if args.max_tokens > 3000:
        args.save_path = str(args.max_tokens)+ "/" +args.model + "_" + args.noise_type + "_" + str(args.n_shot) + "_" + str(args.reverse) + "_" + args.position
        if not os.path.exists(args.log_dir + args.task_name + "/" + str(args.max_tokens)):
            os.makedirs(args.log_dir + args.task_name + "/" + str(args.max_tokens))
    else:
        args.save_path = args.model + "_" + args.noise_type + "_" + str(args.n_shot) + "_" + str(args.reverse) + "_" + args.position
        if not os.path.exists(args.log_dir + args.task_name):
            os.makedirs(args.log_dir + args.task_name)

    log = Logger(args.log_dir + args.task_name + "/" + args.save_path + ".log")

    start = time.time()
    log.logger.info("************************Start Test**********************")
    log.logger.info(
        f"【task】: {args.task_name} 【model】: {args.model} 【noise_type】: {args.noise_type} 【n_shot】: {str(args.n_shot)} 【reverse】: {args.reverse} 【position】: {args.position} 【max_tokens】: {args.max_tokens}\n"
    )
    run(args, log)
    data = load_pkl("predictions/" + args.task_name + "/" + args.save_path + ".pkl")
    index = [i for i in range(args.data_size) if data[i]['prediction'] != '']
    answers = [data[i]['answer'] for i in index]
    preds = [data[i]['prediction'] for i in index]
    print(f'length of preds: {len(preds)}')
    if len(preds) != args.data_size:
        log.logger.info("Warning: Some data is not predicted.")
        log.logger.info("The number of data not predicted: {}".format(args.data_size - len(preds)))
    report = get_eval_report(preds, answers,args)

    log.logger.info("******************Classification Report*****************")
    log.logger.info("\n" + report)
    end = time.time()
    log.logger.info("************************End Test************************")
    log.logger.info("Processing time: {} mins".format((end - start) / 60))
    print(
        f'save log to {args.log_dir + args.task_name + "/" + args.save_path +".log"}'
    )

if __name__ == "__main__":
    main()


from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from collections import Counter
from script.utils import remove_noise

def exact_match(preds, answers):
    correct = 0
    for pred, answer in zip(preds, answers):
        if pred.strip().lower() == answer.strip().lower():
            correct += 1
    return correct / len(answers)



def precision_recall_f1(preds, answers):
    def compute_precision_recall_f1(pred, answer):
        pred_tokens = pred.strip().lower().split()
        answer_tokens = answer.strip().lower().split()
        common_tokens = Counter(pred_tokens) & Counter(answer_tokens)
        num_common = sum(common_tokens.values())

        if num_common == 0:
            return 0, 0, 0

        precision = num_common / len(pred_tokens) if len(pred_tokens) > 0 else 0
        recall = num_common / len(answer_tokens) if len(answer_tokens) > 0 else 0

        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1

    precision_sum = 0
    recall_sum = 0
    f1_sum = 0
    for pred, answer in zip(preds, answers):
        precision, recall, f1 = compute_precision_recall_f1(pred, answer)
        precision_sum += precision
        recall_sum += recall
        f1_sum += f1

    avg_precision = precision_sum / len(answers)
    avg_recall = recall_sum / len(answers)
    avg_f1 = f1_sum / len(answers)

    return avg_precision, avg_recall, avg_f1


def bleu_score(preds, answers):
    bleu_sum = 0
    for pred, answer in zip(preds, answers):
        bleu_sum += sentence_bleu([answer.strip().lower().split()], pred.strip().lower().split())
    return bleu_sum / len(answers)

def rouge_score(preds, answers):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_sum = 0
    for pred, answer in zip(preds, answers):
        score = scorer.score(answer.strip().lower(), pred.strip().lower())
        rouge_sum += score['rougeL'].fmeasure
    return rouge_sum / len(answers)


def get_eval_report(preds, answers,args):
    preds = [remove_noise(pred).lower() for pred in preds]
    answers = [remove_noise(answer).lower() for answer in answers]

    em_score = exact_match(preds, answers)
    print(f"Exact Match Score: {em_score}")

    precision, recall, f1 = precision_recall_f1(preds, answers)
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    bleu = bleu_score(preds, answers)
    print(f"BLEU Score: {bleu}")

    rouge = rouge_score(preds, answers)
    print(f"ROUGE Score: {rouge}")

    report = f"Exact Match: {em_score}\nPrecison: {precision}\nRecall: {recall}\nF1 Score: {f1}\nBLEU Score: {bleu}\nROUGE Score: {rouge}"
    return report
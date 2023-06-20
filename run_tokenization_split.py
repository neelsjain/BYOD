import numpy as np
import torch

import csv
import random
import nltk

import argparse

from BYOD import tokenization_metric
from BYOD.utils import get_dataset, get_model_n_tokenizer, wikitext_detokenizer


def filter_dataset(dataset):
    sentences_filtered = []
    # want to filter out sentences that are too short for these test so test at 50
    dataset = dataset.filter(lambda x: len(x["text"]) >= 50)
    for i in range(len(dataset)):
        example = dataset[i]["text"]
        example = wikitext_detokenizer(example)
        sentences = nltk.sent_tokenize(example)
        sentences = [sent for sent in sentences if len(sent.split()) > 5]
        sentences_filtered = sentences_filtered + sentences
    return sentences_filtered


def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    # get dataset
    dataset = get_dataset(args.dataset_name, args.dataset_config, args.split, args)
    # filter dataset
    dataset = filter_dataset(dataset)
    # get model and tokenizer
    model, tokenizer = get_model_n_tokenizer(args.model_name, args=args)
    print(f"___________{args.num_splits}-Splits___________")
    # get tokenization metric
    JSD_diff, percent_same_tok = tokenization_metric(model, dataset, tokenizer, num_splits=args.num_splits, max_examples=args.max_examples)
    # save the results
    # result_row = [
    #     args.model_name,
    #     args.num_splits,
    #     len(JSD_diff),
    #     percent_same_tok,
    #     np.mean(JSD_diff),
    #     np.std(JSD_diff),
    #     np.median(JSD_diff),
    # ]
    # print(result_row)
    # with open("tokenization_metric/" + args.output_file, "a") as csvfile:
    #     csvwriter = csv.writer(csvfile)
    #     # Model Name, Num Splits, Samples, Percent Same Tokenization, LogPPL Mean, LogPPL Std, LogPPL Median, JSD Mean, JSD Std, JSD Median
    #     csvwriter.writerow(result_row)

    with open("results.csv", mode="a") as file:
        writer = csv.writer(file)
        writer.writerow([args.model_name, "tokenization", len(JSD_diff), np.mean(JSD_diff), np.std(JSD_diff) / np.sqrt(len(JSD_diff))])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt2", help="model name")
    parser.add_argument("--dataset_name", type=str, default="wikitext", help="dataset name")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1", help="dataset config")
    parser.add_argument("--split", type=str, default="train", help="split of the dataset")
    parser.add_argument("--fp16", default=False, type=bool)
    parser.add_argument("--max_examples", type=int, default=1000, help="maximum number of examples to evaluate")
    parser.add_argument("--num_splits", type=int, default=5, help="number of splits")
    parser.add_argument("--output_file", type=str, default="tokenization_results.csv", help="output file")
    parser.add_argument("--cache_dir_dataset", type=str, default="datasets", help="output file")
    parser.add_argument("--cache_dir_model", type=str, default="models", help="output file")
    parser.add_argument("--seed", type=int, default=42, help="seed")
    args = parser.parse_args()
    main(args)

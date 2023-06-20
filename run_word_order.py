import argparse
import torch
import random
import nltk

nltk.download("punkt")

from datasets import load_dataset
import numpy as np

import csv

from BYOD import word_order_metric
from BYOD.utils import get_model_n_tokenizer


def main(args):
    seed = args.seed
    random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", cache_dir=args.cache_dir_dataset).with_format("torch")
    print("Dataset sample: ", dataset["text"][3])

    print("Filtering the dataset")
    filtered_dataset = dataset.filter(lambda example: len(example["text"].split()) > 20)  # filter out short sentences
    model, tokenizer = get_model_n_tokenizer(args.model_name, args=args)

    sens_score, sens_ste, model_sensivity_scores = word_order_metric(model, dataset, tokenizer, n_swap=args.n_swap, max_examples=args.max_examples, data_cleaned=False)

    # results_row = [
    #     args.model_name,
    #     len(model_sensivity_scores),
    #     args.n_swap,
    #     np.mean(model_sensivity_scores),
    #     np.std(model_sensivity_scores),
    #     np.median(model_sensivity_scores),
    #     np.min(model_sensivity_scores),
    #     np.max(model_sensivity_scores),
    # ]
    # print("Results Row: ", results_row)
    # with open("word_order/word_order_results.csv", mode="a") as file:
    #     writer = csv.writer(file)
    #     writer.writerow(results_row)

    with open("results.csv", mode="a") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                args.model_name,
                "word order",
                len(model_sensivity_scores),
                np.median(model_sensivity_scores),
                np.std(model_sensivity_scores) / np.sqrt(len(model_sensivity_scores)),
            ]
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="gpt2")
    parser.add_argument("--max_examples", default=5000, type=int)
    parser.add_argument("--n_swap", default=1, type=int)
    parser.add_argument("--fp16", default=False, type=bool)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--without_replacement", action="store_true")
    parser.add_argument("--cache_dir_model", default="models")
    parser.add_argument("--cache_dir_dataset", default="datasets")
    args = parser.parse_args()

    print(args)
    main(args)

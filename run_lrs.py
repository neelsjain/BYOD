import torch
import argparse

import random
import numpy as np

import csv
from BYOD import lrs_metric
from BYOD.utils import get_model_n_tokenizer


def main(args):
    torch.manual_seed(args.set_seed)
    torch.cuda.manual_seed(args.set_seed)
    random.seed(args.set_seed)
    np.random.seed(args.set_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    # from transformers import AutoTokenizer, AutoModelForCausalLM
    model, tokenizer = get_model_n_tokenizer(args.model_name, args=args)

    print("Loading dataset wikitext data")
    from datasets import load_dataset

    # use train because it is bigger
    wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="train").with_format("torch")
    wiki = wiki.filter(lambda example: len(example["text"].split()) > 100)

    lrs_mean, lrs_stderr, logits_diff = lrs_metric(
        model, wiki, tokenizer, args.num_sentences_input, args.num_sentences_swap, args.max_examples
    )

    # result_row = [
    #     args.model_name,
    #     len(logits_diff),
    #     np.mean(logits_diff),
    #     np.std(logits_diff),
    #     np.median(logits_diff),
    #     args.dataset_name,
    #     args.set_seed,
    #     args.num_sentences_input,
    # ]
    # print(result_row)

    # with open("context_sensitivity/lrs_results.csv", mode="a") as file:
    #     writer = csv.writer(file)
    #     writer.writerow(result_row)

    with open("results.csv", mode="a") as file:
        writer = csv.writer(file)
        writer.writerow(
            [args.model_name, "context", len(logits_diff), np.mean(logits_diff), np.std(logits_diff) / np.sqrt(len(logits_diff))]
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--dataset_name", type=str, default="wiki", help="wiki")
    parser.add_argument("--num_sentences_input", type=int, default=3, help="Number of sentences in input")
    parser.add_argument("--num_sentences_swap", type=int, default=2, help="Number of sentences in input")
    parser.add_argument("--max_examples", type=int, default=1000)
    parser.add_argument("--set_seed", type=int, default=42)
    parser.add_argument("--fp16", default=False, type=bool)
    parser.add_argument("--cache_dir_model", type=str, default="models")
    parser.add_argument("--cache_dir_dataset", type=str, default="datasets")
    args = parser.parse_args()
    main(args)

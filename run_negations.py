import torch

import csv

from BYOD.utils import WikiDataset, get_model_n_tokenizer
from BYOD import negation_metric

#  DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):

    wiki_simple = []
    # no prompts were used; however, the code is left here for future use
    prompt = ""
    prompt_end = ""
    # # open file and read the content in a list
    if args.load_dataset == "wiki_topic":
        # Note wiki simple is used here for cleaner sentences and easier to grab the topic sentence
        wiki_simple = WikiDataset(
            corpus_path="wikipedia",
            corpus_name="20220301.simple",
            topic_sentence=True,
            all_sentences=False,
            max_examples=args.max_examples * 3,
            cache_dir=args.cache_dir_dataset,
            seed=args.seed,
        ).get_dataset()
    else:
        raise Exception("Invalid load_dataset name")

    print("Downloading from Huggingface")
    model_name = args.model_name
    model, tokenizer = get_model_n_tokenizer(args.model_name, args=args, low_cpu_mem_usage=True)
    model.eval()
    mean_loss_diff, std_err_loss_diff, scores = negation_metric(
        model,
        wiki_simple,
        tokenizer,
        prompt,
        prompt_end,
        max_examples=args.max_examples,
    )

    # result_row = [
    #     args.model_name,
    #     args.max_examples,
    #     np.round(mean_loss_diff, 4),
    #     np.round(std_err_loss_diff, 4),
    #     mean_output_loss,
    #     std_output_loss,
    #     args.load_dataset,
    # ]
    # print(result_row)
    # with open("negation_results.csv", mode="a") as file:
    #     writer = csv.writer(file)
    #     # model_name, mean_loss_diff, std_err_loss_diff, mean_output_loss, std_output_loss, percent_sign_wrong_way, max_examples, load_dataset
    #     writer.writerow(result_row)

    with open("results.csv", mode="a") as file:
        writer = csv.writer(file)
        writer.writerow([args.model_name, "negations", args.max_examples, mean_loss_diff, std_err_loss_diff])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--load_dataset", type=str, default="wiki_topic")
    parser.add_argument("--max_examples", type=int, default=1000)
    parser.add_argument("--fp16", default=False, type=bool)
    parser.add_argument("--cache_dir_model", type=str, default="models")
    parser.add_argument("--cache_dir_dataset", type=str, default="datasets")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)

import numpy as np
import torch
import random
import nltk
from .utils.JSD import JSD
from .utils.hf_utils import wikitext_detokenizer

_default_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def tokenization_metric(model, dataset, tokenizer, num_splits=5, max_examples=1000, device=_default_device):
    tokenization_pairs, percent_same_tok = create_tokenization_pairs(dataset, tokenizer, num_splits=num_splits, max_examples=max_examples)

    JSD_diff = []
    for i in range(len(tokenization_pairs)):
        org_tokenization = tokenization_pairs[i][0]
        chopped_tokenization = tokenization_pairs[i][1]
        with torch.no_grad():
            # Note not batched to eliminate any padding effects
            org_output = model(**org_tokenization.to(device), labels=org_tokenization["input_ids"])
            chopped_output = model(**chopped_tokenization.to(device), labels=chopped_tokenization["input_ids"])
            org_logits = torch.nn.Softmax(dim=-1)(org_output["logits"][0][-1]).to(torch.float32)
            transformed_logits = torch.nn.Softmax(dim=-1)(chopped_output["logits"][0][-1]).to(torch.float32)
            JSD_diff.append(JSD(org_logits + 1e-14, transformed_logits + 1e-14).item())
    return JSD_diff, percent_same_tok


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


def chop_string_every_n_characters(text, num_splits):
    length = len(text)
    last_idx = None
    text_splits = []
    indicies = np.arange(0, length, num_splits).tolist()
    if length not in indicies:
        indicies.append(length)
    assert indicies[-1] == length
    assert len(indicies) == len(set(indicies))
    for i, idx in enumerate(np.arange(0, length, num_splits).tolist() + [len(text)]):
        if i == 0:
            last_idx = idx
            continue
        else:
            text_splits.append(text[last_idx:idx])
            last_idx = idx

    return text, text_splits


def create_tokenization_for_chopped_pieces(chopped_text, tokenizer):
    # we will tokenize each chopped piece and combine them into one tokenized piece
    for i, chopped_text_piece in enumerate(chopped_text):
        tokenized_chopped_text_piece = tokenizer(chopped_text_piece, return_tensors="pt")
        if i == 0:
            combined_tokenized_chopped_text = tokenized_chopped_text_piece
        else:
            if "llama" in tokenizer.name_or_path.lower():
                # Hack for llama including bos token in input_ids and extra whitespace at the beginning; there may be a better way to do this could not find the sentencepiece argument; please create a pull request if you know or find it
                tokenized_chopped_text_piece = tokenizer("=" + chopped_text_piece, return_tensors="pt")
                # This probably can can get looped over by picking a different token to start with and checking if it is the same as the first token
                # However, we leave this as is because it is more deterministic, and thus, more clear where the ignored samples may be coming from
                if tokenized_chopped_text_piece["input_ids"][:, 1][0].item() != 353:
                    print("Gonna Try Another Token")
                    print(tokenized_chopped_text_piece["input_ids"])
                    print(chopped_text_piece)
                    print("=" + chopped_text_piece)
                    # TRY ANOTHER TOKEN
                    tokenized_chopped_text_piece = tokenizer("THE" + chopped_text_piece, return_tensors="pt")
                    if tokenized_chopped_text_piece["input_ids"][:, 1][0].item() != 6093:
                        print("Ignoring this sample")
                        print(tokenized_chopped_text_piece["input_ids"])
                        print(chopped_text_piece)
                        print("THE" + chopped_text_piece)
                        print("Ignoring this sample")

                tokenized_chopped_text_piece["input_ids"] = tokenized_chopped_text_piece["input_ids"][:, 2:]
                tokenized_chopped_text_piece["attention_mask"] = tokenized_chopped_text_piece["attention_mask"][:, 2:]

            for key in tokenized_chopped_text_piece:
                combined_tokenized_chopped_text[key] = torch.cat(
                    (combined_tokenized_chopped_text[key], tokenized_chopped_text_piece[key]), dim=1
                )

    return combined_tokenized_chopped_text


def create_tokenization_pairs(dataset, tokenizer, num_splits=2, max_examples=-1):
    tokenization_pairs = []
    count_same_tok = 0
    random.shuffle(dataset)
    for i in range(len(dataset)):
        text = dataset[i]
        if i % 1000 == 0:
            print(text)
        if max_examples != -1 and i == max_examples:
            break
        original_text, chopped_text = chop_string_every_n_characters(text, num_splits)
        org_tokenization = tokenizer(original_text, return_tensors="pt")
        chopped_tokenization = create_tokenization_for_chopped_pieces(chopped_text, tokenizer)
        if org_tokenization["input_ids"].shape == chopped_tokenization["input_ids"].shape:
            # need to nest this because if mismatched sized can't compare if the tokenization did not change
            if (org_tokenization["input_ids"] == chopped_tokenization["input_ids"]).all():
                count_same_tok += 1

        tokenization_pairs.append(
            [tokenizer(original_text, return_tensors="pt"), create_tokenization_for_chopped_pieces(chopped_text, tokenizer)]
        )
        if i % 1000 == 0:
            print(tokenizer(original_text, return_tensors="pt"))
            print(create_tokenization_for_chopped_pieces(chopped_text, tokenizer))
            print(count_same_tok / (i + 1))
    if i == 0:
        i += 1
    return tokenization_pairs, count_same_tok / len(tokenization_pairs)

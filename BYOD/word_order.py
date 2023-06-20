import torch
import random
import nltk

nltk.download("punkt")

from tqdm.autonotebook import tqdm
import numpy as np
from .utils.JSD import JSD
from .utils.hf_utils import wikitext_detokenizer

_default_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def word_order_metric(model, dataset, tokenizer, n_swap=1, max_examples=1000, device=_default_device, data_cleaned=True):
    data_cleaned = data_cleaned
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Filtering the dataset")
    try: 
        filtered_dataset = dataset.filter(lambda example: len(example["text"].split()) > 20)  # filter out short sentences
    except: # if dataset is a list of strings
        import datasets
        import pandas as pd
        hf_dataset = datasets.Dataset.from_pandas(pd.DataFrame(data={'text': dataset}))
        filtered_dataset = hf_dataset.filter(lambda example: len(example["text"].split()) > 20)
    results_row = []
    print("Dataset sample: ", filtered_dataset["text"][3])
    print("N Swap: ", n_swap)
    n_swapped_dataset = filtered_dataset.map(swap_words_in_sentence, fn_kwargs={"n": n_swap, "data_cleaned": data_cleaned})
    if n_swap == 0:
        new_n_swap_pair = list(zip(n_swapped_dataset["text"], n_swapped_dataset["text"]))
    else:
        new_n_swap_pair = list(zip(n_swapped_dataset["text"], n_swapped_dataset["swapped"]))
    random.shuffle(new_n_swap_pair)
    if max_examples != -1:
        new_n_swap_pair = new_n_swap_pair[:max_examples]
    print("new_n_swap_pair: ", new_n_swap_pair[:5])
    model.eval()
    model_sensivity_scores = get_sent_pair_sens_score(new_n_swap_pair, model, tokenizer, device=device)
    return np.median(model_sensivity_scores), \
                np.std(model_sensivity_scores) / np.sqrt(len(model_sensivity_scores)), \
                model_sensivity_scores


def get_sent_pair_sens_score(pairs, model, tokenizer, device=_default_device):
    similarity_sensivity_scores = []
    i = 0
    for pair in tqdm(pairs):
        i += 1
        element_0 = pair[0]
        element_1 = pair[1]
        inputs_0 = tokenizer(element_0, padding=True, truncation=True, return_tensors="pt")
        inputs_1 = tokenizer(element_1, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            # Note not batched to eliminate any padding effects
            outputs_0 = model(**inputs_0.to(device), labels=inputs_0["input_ids"])
            outputs_1 = model(**inputs_1.to(device), labels=inputs_1["input_ids"])

        # This is a hack for fp16 compatibility; future version might just use log probs in JSD
        logits_org = torch.nn.Softmax(dim=-1)(outputs_0["logits"][0][-1]).to(torch.float32)
        logits_transformed = torch.nn.Softmax(dim=-1)(outputs_1["logits"][0][-1]).to(torch.float32)

        similarity_sensivity_scores.append(JSD(logits_org + 1e-14, logits_transformed + 1e-14).item())

        if i == 1:
            print("JSD: ", JSD(logits_org + 1e-14, logits_transformed + 1e-14).item())

    return similarity_sensivity_scores


def swap_words_in_sentence(example, n=1, data_cleaned=True):
    # Split the paragraph into sentences
    sentences = nltk.sent_tokenize(example["text"])

    # remove sentences with less than 4 words
    sentences = [sent for sent in sentences if len(sent.split()) > 5]

    # Choose a random sentence
    sentence = random.choice(sentences)

    if n == 0:
        if not data_cleaned: sentence = wikitext_detokenizer(sentence)
        return {"swapped": sentence, "text": sentence}
    else:
        # Find the longest sentence
        if not data_cleaned: 
            sentence = wikitext_detokenizer(sentence)
        # Tokenize the longest sentence
        token = nltk.word_tokenize(sentence)

        for i in range(n):
            # Choose two random indices
            idx1, idx2 = random.sample(range(len(token)), 2)

            # Swap the words at the two indices
            token[idx1], token[idx2] = token[idx2], token[idx1]
        # Reconstruct the modified sentence
        modified_sent = " ".join(token)

        return {"swapped": modified_sent, "text": sentence}

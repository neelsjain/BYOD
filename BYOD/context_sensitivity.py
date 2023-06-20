import torch
import nltk
import random
import numpy as np

from .utils import JSD, wikitext_detokenizer

_default_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def lrs_metric(model, data, tokenizer, num_sentences_input=3, num_sentences_swap=2, max_examples=1000, device=_default_device):
    def get_triplets(data):
        last_sentences = []
        c_sentences = []
        t_sentences = []
        random.shuffle(data)
        for i, element in enumerate(data):
            element = wikitext_detokenizer(element)
            if "\n" in element[-10:]:
                element = element.replace("\n", "").strip()
            else:
                element = element.strip()
            element_sentence = nltk.sent_tokenize(element)

            if len(element_sentence) < num_sentences_input + num_sentences_swap + 1:
                continue

            if i == 0:
                new_sentences = nltk.sent_tokenize(data[-1])
                sentences_to_add = new_sentences[:num_sentences_swap]
            else:
                new_sentences = nltk.sent_tokenize(data[i - 1])
                sentences_to_add = new_sentences[:num_sentences_swap]

            last_sentences.append(element_sentence[-1])
            c_sentences.append(element_sentence[-num_sentences_input:])
            t_sentences.append(sentences_to_add + element_sentence[-(num_sentences_input - num_sentences_swap) :])

        return last_sentences, c_sentences, t_sentences

    triplets_ = get_triplets(data["text"])
    print("Number of examples: ", len(triplets_[0]))
    if max_examples < len(triplets_[0]) and max_examples != -1:
        triplets = [triplets_[0][:max_examples], triplets_[1][:max_examples], triplets_[2][:max_examples]]
        print("New number of examples: ", len(triplets[0]))
    else:
        triplets = triplets_

    logits_diff = []

    for i, (last_sentence, c_sentences, t_sentences) in enumerate(zip(triplets[0], triplets[1], triplets[2])):
        last_sentence_updated = " " + last_sentence
        c_sentences = " ".join(c_sentences)
        t_sentences = " ".join(t_sentences)
        last_sentence_encoded = tokenizer.encode(last_sentence_updated, return_tensors="pt")
        position_slice = len(last_sentence_encoded[0])
        # get rid of small sentences
        if position_slice < 2:
            print("Last Sentence: ")
            print(last_sentence_updated)
            continue

        batch_c = tokenizer(c_sentences, return_tensors="pt", padding=False).to(device)
        batch_t = tokenizer(t_sentences, return_tensors="pt", padding=False).to(device)

        if i == 0:
            print("First Example: ", c_sentences)
            print("Second Example: ", t_sentences)

        with torch.no_grad():
            outputs_c = model(**batch_c, labels=batch_c["input_ids"], output_hidden_states=True)
            outputs_t = model(**batch_t, labels=batch_t["input_ids"], output_hidden_states=True)
            # offset for predicted token
            logits_c = outputs_c.logits[0][-(position_slice + 1) : -1]  # sentence x vocab
            logits_t = outputs_t.logits[0][-(position_slice + 1) : -1]  # sentence x vocab

            diff = (
                JSD(
                    torch.nn.Softmax(dim=-1)(logits_c).to(torch.float32) + 1e-14,
                    torch.nn.Softmax(dim=-1)(logits_t).to(torch.float32) + 1e-14,
                )
                .mean()
                .item()
            )
            logits_diff.append(diff)

            if i % 1000 == 0:
                print("JSD: ", diff)

    return np.mean(logits_diff), np.std(logits_diff) / np.sqrt(len(logits_diff)), logits_diff

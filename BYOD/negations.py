import numpy as np
import torch

_default_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def negation_metric(model, data, tokenizer, prompt="", prompt_end="", max_examples=1000, device=_default_device):
    # Filter data
    filter_data = filter_dataset(data)
    dataset_transformed = apply_transformation(filter_data)
    assert len(filter_data) == len(dataset_transformed)
    if len(dataset_transformed) > max_examples:
        dataset_transformed = dataset_transformed[:max_examples]
        filter_wiki = filter_data[:max_examples]
    else:
        print("Not enough examples, using all")
        filter_wiki = filter_data

    loss_diff = []
    output_loss = []
    for element, element_transformed in zip(filter_wiki, dataset_transformed):
        element = prompt + element + prompt_end
        element_transformed = prompt + element_transformed + prompt_end
        input_encoded = tokenizer(element, return_tensors="pt", truncation=True, max_length=128).to(device)
        input_encoded_transformed = tokenizer(element_transformed, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            try:
                # we do not batch as this affects smaller models like gpt2 with the absolute position embeddings -- padding behave weirdly here
                outputs = model(**input_encoded, labels=input_encoded["input_ids"])
                outputs_transformed = model.forward(**input_encoded_transformed, labels=input_encoded_transformed["input_ids"])
                output_loss.append(outputs.loss.item())
                loss_diff.append(outputs_transformed.loss.item() - outputs.loss.item())
            except Exception as e:
                print(f"Error {e}")
                print(element)
                print(tokenizer(element, return_tensors="pt"))
                print(element_transformed)
                continue

    return (
        np.array(loss_diff).mean(),
        np.std(loss_diff) / np.sqrt(len(loss_diff)),
        loss_diff
        # np.mean(output_loss),
        # np.std(output_loss),
    )


def filter_dataset(dataset):
    """
    filters the dataset for if there is a ``is'', ``was'', etc
    """
    dataset_filter = []
    for i, element in enumerate(dataset):
        if " is " in element and " is not " not in element:
            dataset_filter.append(element)
        elif " was " in element and " was not " not in element:
            dataset_filter.append(element)
        elif " are " in element and " are not " not in element:
            dataset_filter.append(element)
        elif " were " in element and " were not " not in element:
            dataset_filter.append(element)

    return dataset_filter


def apply_transformation(dataset):
    """
    filters the dataset for if there is a ``is'', ``was'', etc.
    """
    dataset_transformed = []
    for i, element in enumerate(dataset):
        if " is " in element and " is not " not in element:
            dataset_transformed.append(element.replace(" is ", " is not ", 1))
        elif " was " in element and " was not " not in element:
            dataset_transformed.append(element.replace(" was ", " was not ", 1))
        elif " are " in element and " are not " not in element:
            dataset_transformed.append(element.replace(" are ", " are not ", 1))
        elif " were " in element and " were not " not in element:
            dataset_transformed.append(element.replace(" were ", " were not ", 1))
    return dataset_transformed

"""Utilities to load models and data."""
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from datasets import load_dataset
import torch
import re


def get_dataset(dataset_name, dataset_config, split, args=None):
    dataset = load_dataset(dataset_name, dataset_config, cache_dir=args.cache_dir_dataset)
    dataset = dataset[split]
    return dataset


def get_model_n_tokenizer(model_name, args=None, trust_remote_code=True, low_cpu_mem_usage=False):
    if "llama" in model_name:
        print("Loading LLAMA model")
        model, tokenizer = llama_loading(model_name, args=args)
    elif args.fp16:
        print("Loading FP16 model")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=low_cpu_mem_usage,
                torch_dtype=torch.float16,
                cache_dir=args.cache_dir_model,
            )
        except Exception as e:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch.float16,
                cache_dir=args.cache_dir_model,
            ).cuda()
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left", cache_dir=args.cache_dir_model)
        tokenizer.pad_token = tokenizer.eos_token
    else:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                device_map="auto",
                trust_remote_code=trust_remote_code,
                low_cpu_mem_usage=low_cpu_mem_usage,
                cache_dir=args.cache_dir_models,
            )
        except Exception as e:
            model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True, cache_dir=args.cache_dir_model).cuda()
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir_model)
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def llama_loading(model_name, args=None):
    if args.fp16:
        print("Loading FP16 model")
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True).cuda()
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
    tokenizer = LlamaTokenizer.from_pretrained(model_name, padding_side="left")
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))
    model.eval()
    return model, tokenizer


def wikitext_detokenizer(string):
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    string = string.replace(r"\'", "'")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")

    return string

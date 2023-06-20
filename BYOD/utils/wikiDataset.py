import datasets
import nltk
import itertools
import random
import re


class WikiDataset:
    def __init__(
        self,
        corpus_path="wikitext",
        corpus_name="wikitext-2-raw-v1",
        topic_sentence=True,
        all_sentences=False,
        cache_dir=None,
        max_examples=-1,
        seed=42,
    ) -> None:
        self.topic_sentence = topic_sentence
        self.all_sentences = all_sentences
        self.max_examples = max_examples
        self.seed = seed
        self.cache = cache_dir
        self.corpus_path = corpus_path
        self.corpus_name = corpus_name
        if self.all_sentences and self.topic_sentence:
            raise ValueError("Can't have both topic_sentence and all_sentences")

    def wikitext_detokenizer(self, string):
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

    def download_dataset(self, huggingface_hub=True):
        if huggingface_hub:
            # Use simple because data is cleaner and is more than size needed for this test than original wikipedia, although it should not matter which wikipedia we use
            dataset = datasets.load_dataset(self.corpus_path, self.corpus_name, cache_dir=self.cache)
        else:
            raise NotImplementedError("Only huggingface hub is supported")
        return dataset

    def get_dataset(self):
        random.seed(self.seed)
        dataset = self.download_dataset()
        dataset = dataset["train"]
        if self.topic_sentence:
            # get text
            if self.max_examples != -1 and self.max_examples < len(dataset["text"]):
                dataset_text = dataset["text"]
                print("Shuffling dataset")
                random.shuffle(dataset_text)
                print("Slicing dataset total examples: ", self.max_examples)
                dataset = dataset_text[: self.max_examples]
                print("Done slicing dataset")
            else:
                dataset = dataset["text"]
            # split into sentences
            dataset = list(filter(lambda x: len(x) > 1, dataset))  # filter out empty strings
            dataset = list(map(lambda x: nltk.tokenize.sent_tokenize(self.wikitext_detokenizer(x))[0], dataset))

        elif self.all_sentences:
            # get text
            dataset = dataset["text"]
            # split into sentences
            print("Sentence Tokenizing")
            dataset = list(filter(lambda x: len(x) > 1, dataset))  # filter out empty strings
            dataset = list(map(lambda x: nltk.tokenize.sent_tokenize(self.wikitext_detokenizer(x)), dataset))
            print("Flattening")
            dataset = list(itertools.chain.from_iterable(dataset))
            # filter out empty strings
            # dataset = [x for x in dataset if x != ""]
            # remove the sentences that are too long (more than 2000 characters)
            dataset = [x for x in dataset if len(x) <= 2000]

            if self.max_examples != -1 and self.max_examples < len(dataset):
                dataset = random.sample(dataset, self.max_examples)

        else:
            raise NotImplementedError("Only topic_sentence and all_sentences are supported")

        return dataset

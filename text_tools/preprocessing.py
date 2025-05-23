# /usr/bin/env python3
# -*- coding: utf-8 -*-

import gc
import string
import logging
from typing import List

from tqdm import tqdm
from gensim.models.phrases import Phrases

import spacy

# set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_spacy_model(lang: str, disable: List[str] = []):
    """Load the appropriate spaCy model for the specified language"""
    try:
        if lang == "en":
            nlp = spacy.load("en_core_web_sm", disable=disable)
        elif lang == "de":
            nlp = spacy.load("de_core_news_sm", disable=disable)
        elif lang == "fr":
            nlp = spacy.load("fr_core_news_sm", disable=disable)
        elif lang == "it":
            nlp = spacy.load("it_core_news_sm", disable=disable)
        else:
            raise ValueError(f"Unsupported language: {lang}")
    except OSError:
        raise OSError(
            f"Language model for '{lang}' not found. "
            f"Please install the appropriate spaCy model. "
            f"Example: `python -m spacy download {lang}_core_news_sm`"
        )
    return nlp


class MWEParser:
    def __init__(
        self,
        lang: str = "en",
        connector_words: List[str] = None,
        min_count: int = 4,
        threshold: float = 0.85,
        scoring: str = "npmi",
    ):
        self.connector_words = set(connector_words) if connector_words else set()
        self.connector_words.update(string.punctuation)

        self.scoring = scoring
        self.min_count = min_count
        self.threshold = threshold

        self.nlp = load_spacy_model(lang, disable=["ner", "textcat", "tagger"])
        self.phraser = None

    def learn_phraser(self, texts: List[str]) -> Phrases:
        # learn phrases using Gensim
        bigram_model = Phrases(
            min_count=self.min_count,
            threshold=self.threshold,
            scoring=self.scoring,
            connector_words=self.connector_words,
        )

        trigram_model = Phrases(
            min_count=self.min_count,
            threshold=self.threshold,
            scoring=self.scoring,
            connector_words=self.connector_words,
        )

        # Stream over chunks, extract sentences, and update the phrase models
        for text in tqdm(texts, desc="Learning phrases..."):
            for doc in self.nlp.pipe([text], n_process=1):
                doc_sent_tokens = [
                    [
                        token.text
                        for token in sent
                        if not (
                            token.is_digit
                            or token.like_num
                            or token.is_punct
                            or token.is_space
                            or token.text.startswith("|")
                        )
                    ]
                    for sent in doc.sents
                ]
                doc_sent_tokens = [
                    sent for sent in doc_sent_tokens if len(sent) > 0
                ]  # filter out empty sentences
                bigram_model.add_vocab(doc_sent_tokens)
                trigram_model.add_vocab(bigram_model[doc_sent_tokens])

        bigram_model.freeze()
        trigram_model.freeze()

        # clean up
        del bigram_model
        gc.collect()

        self.phraser = trigram_model
        logger.info("Phraser model learned successfully.")

    def extract_phrases(self) -> List[str]:
        return [phrase.replace("_", " ") for phrase in self.phraser.export_phrases()]

    def save_to_disk(self, path: str):
        """
        Save the learned phrase model to disk.

        Note, the model is saved in a pickle format.

        Args:
            path (str): The path to save the model.
        """
        if not path.endswith(".pkl"):
            path += ".pkl"

        if self.phraser is None:
            raise ValueError(
                "No phrase model learned. Please call learn_phraser() first."
            )

        logger.info(f"Saving learned phrase model to {path}")
        self.phraser.save(path)

    def load_from_disk(self, path: str):
        """
        Load a learned phrase model from disk.

        Args:
            path (str): The path to load the model from.
        """

        if not path.endswith(".pkl"):
            raise ValueError(f"Expected a .pkl file, got {path}")

        if self.phraser is not None:
            logger.warning("Overwriting existing phrase model.")

        logger.info(f"Loading learned phrase model from {path}")
        self.phraser = Phrases.load(path)


class PhrasalTokenizer:
    def __init__(
        self,
        lang: str,
        disable: List[str] = None,
        mwes: List[str] = None,
        concat_token: str = " ",
        stop_words: List[str] = None,
        keep_num: bool = False,
        keep_punct: bool = False,
        keep_space: bool = False,
        keep_email: bool = False,
        keep_url: bool = False,
        keep_stopwords: bool = False,
        lower: bool = True,
    ):
        self.lang = lang
        self.disable = disable if disable else []
        self.nlp = load_spacy_model(lang, disable=self.disable)

        self.mwes = mwes if mwes else []
        self._add_mwe_patterns()

        self.concat_token = concat_token

        self.stop_words = stop_words if stop_words else []
        self.nlp.Defaults.stop_words.update(self.stop_words)

        self.keep_num = keep_num
        self.keep_punct = keep_punct
        self.keep_space = keep_space
        self.keep_email = keep_email
        self.keep_url = keep_url
        self.keep_stopwords = keep_stopwords
        self.lower = lower

    def _add_mwe_patterns(self):
        """Add multi-word expression patterns to the spaCy pipeline"""
        if len(self.mwes) > 0 and "entity_ruler" not in self.nlp.pipe_names:
            ruler = self.nlp.add_pipe("entity_ruler")
            for mwe in self.mwes:
                # print(f"Adding pattern: {mwe}")
                ruler.add_patterns([{"label": "MWE", "pattern": mwe}])

    def is_valid_token(self, token):
        # Check if the token is not a stop word and not punctuation
        if not self.keep_stopwords and token.is_stop:
            return False

        # Check if the token is not a punctuation mark
        if not self.keep_punct and (
            token.is_punct
            or token.text in string.punctuation
            or token.text.startswith("|")
        ):
            # NOTE: token.text.startswith("|") is a fix for md tables
            return False

        # Check if the token is not a digit or like a number
        if not self.keep_num and (token.is_digit or token.like_num):
            return False

        # Check if the token is not a space or starts with '|'
        if not self.keep_space and (token.is_space):
            return False

        if not self.keep_url and token.like_url:
            return False

        if not self.keep_email and token.like_email:
            return False

        return True

    def tokenize(self, text):
        """
        Given a text, tokenize it into words and phrases.

        Args:
            text (str): The input text to tokenize.

        Returns:
            List[str]: A list of tokens, including multi-word expressions.
        """
        doc = self.nlp(text)
        with doc.retokenize() as retokenizer:
            for ent in doc.ents:
                retokenizer.merge(
                    doc[ent.start : ent.end],
                    attrs={"LEMMA": ent.text.replace(" ", "_")},
                )

        return [
            x.text.replace(" ", self.concat_token).lower()
            if self.lower
            else x.text.replace(" ", self.concat_token)
            for x in doc
            if self.is_valid_token(x)
        ]


if __name__ == "__main__":

    # Example usage
    from text_tools.constants import CONNECTOR_WORDS
    from text_tools.data import ChunkedDataset

    mwe_parser = MWEParser(
        lang="de",
        connector_words=CONNECTOR_WORDS.get("de", []),
    )

    input_dir = "tests/data/Jahresberichte_sample"
    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    max_tokens = 32

    dataset = ChunkedDataset(
        input_dir=input_dir,
        model_id=model_id,
        max_tokens=max_tokens,
        recursive=False,
    )

    mwe_parser.learn_phraser(dataset["text"])

    mwes = mwe_parser.extract_phrases()
    print(f"Found {len(mwes)} multi-word expressions")
    print(f"First 10 multi-word expressions: {mwes[:10]}")

    assert len(mwes) > 0, "No multi-word expressions found"

    tokenizer = PhrasalTokenizer(lang="de", mwes=mwes, concat_token="_", lower=False)

    tokenizer.tokenize("Der Fotograf Jos Schmit hat für Dies academicus gearbeitet.")
    # ['Fotograf', 'Jos_Schmit', 'Dies_academicus', 'gearbeitet']

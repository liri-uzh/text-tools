# /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multilingual extension of the MWE preprocessing script.
Uses spaCy's multilingual model for language-agnostic processing.

Installation:
    pip install spacy gensim tqdm
    python -m spacy download xx_ent_wiki_sm
"""

import gc
import string
import logging
from typing import Dict, Optional, Set, Union

from tqdm import tqdm
from gensim.models.phrases import Phrases

import spacy

# set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_spacy_model(lang: str, disable: Optional[list[str]] = None):
    """Load the appropriate spaCy model for the specified language"""
    if disable is None:
        disable = []
    try:
        if lang == "multi" or lang == "xx":
            # Use multilingual model for mixed-language support
            # ensure that the model is installed: python -m spacy download xx_ent_wiki_sm
            nlp = spacy.load("xx_ent_wiki_sm", disable=disable)
        elif lang == "en":
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
            f"For multilingual support: `python -m spacy download xx_ent_wiki_sm`"
        )
    return nlp


class MWEParser:
    def __init__(
        self,
        lang: str = "en",
        connector_words: Optional[Set[str]] = None,
        min_count: int = 4,
        threshold: float = 0.85,
        scoring: str = "npmi",
    ):
        """
        Multi-Word Expression Parser.

        Args:
            lang: Language code ('en', 'de', 'fr', 'it', or 'multi' for multilingual)
            connector_words: Set of connector words (language-agnostic when using 'multi')
            min_count: Minimum count for phrase detection
            threshold: Threshold for phrase detection
            scoring: Scoring method ('npmi', 'npmi_sqrt', 'default')
        """
        self.lang = lang
        self.connector_words = connector_words if connector_words else set()
        self.connector_words.update(string.punctuation)

        self.scoring = scoring
        self.min_count = min_count
        self.threshold = threshold

        self.nlp = load_spacy_model(lang, disable=["ner", "textcat", "tagger"])
        if "sentencizer" not in self.nlp.pipe_names:
            self.nlp.add_pipe("sentencizer")
        self.phraser = None

        if lang in ["multi", "xx"]:
            logger.info("Using multilingual model - supports mixed-language documents")

    def learn_phraser(self, texts: list[str]) -> None:
        """
        Learn multi-word expressions from texts.

        Args:
            texts: List of input texts (can be in different languages if lang='multi')
        """
        # learn phrases using Gensim
        bigram_model = Phrases(
            min_count=self.min_count,
            threshold=self.threshold,
            scoring=self.scoring,
            connector_words=frozenset(self.connector_words),
        )

        trigram_model = Phrases(
            min_count=self.min_count,
            threshold=self.threshold,
            scoring=self.scoring,
            connector_words=frozenset(self.connector_words),
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

    def extract_phrases(self) -> list[str]:
        """Extract learned phrases as a list."""
        if self.phraser is None:
            raise ValueError("No phraser learned. Call learn_phraser() first.")
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
        disable: Optional[list[str]] = None,
        mwes: Optional[Set[str]] = None,
        concat_token: str = " ",
        stop_words: Optional[Set[str]] = None,
        keep_num: bool = False,
        keep_punct: bool = False,
        keep_space: bool = False,
        keep_email: bool = False,
        keep_url: bool = False,
        keep_stopwords: bool = False,
        lower: bool = True,
    ):
        """
        Phrasal tokenizer with multi-word expression support.

        Args:
            lang: Language code ('en', 'de', 'fr', 'it', or 'multi' for multilingual)
            disable: Pipeline components to disable
            mwes: Set of multi-word expressions to recognize
            concat_token: Token to use for concatenating MWEs
            stop_words: Additional stopwords to add
            keep_num: Keep numeric tokens
            keep_punct: Keep punctuation
            keep_space: Keep space tokens
            keep_email: Keep email addresses
            keep_url: Keep URLs
            keep_stopwords: Keep stopwords
            lower: Lowercase tokens
        """
        self.lang = lang
        self.disable = disable if disable else []
        self.nlp = load_spacy_model(lang, disable=self.disable)

        self.mwes = mwes if mwes else []
        self._add_mwe_patterns()

        self.concat_token = concat_token

        # # Enhanced stopword handling for multilingual
        # if lang in ["multi", "xx"]:
        #     # Add common stopwords for major languages since multilingual model has limited support
        #     common_stopwords = {
        #         # English
        #         "the",
        #         "a",
        #         "an",
        #         "and",
        #         "or",
        #         "but",
        #         "in",
        #         "on",
        #         "at",
        #         "to",
        #         "for",
        #         "of",
        #         "with",
        #         "by",
        #         "over",
        #         "is",
        #         "are",
        #         "was",
        #         "were",
        #         "be",
        #         "been",
        #         "being",
        #         "have",
        #         "has",
        #         "had",
        #         "do",
        #         "does",
        #         "did",
        #         "will",
        #         "would",
        #         "could",
        #         "should",
        #         "may",
        #         "might",
        #         "can",
        #         "this",
        #         "that",
        #         "these",
        #         "those",
        #         "i",
        #         "you",
        #         "he",
        #         "she",
        #         "it",
        #         "we",
        #         "they",
        #         "me",
        #         "him",
        #         "her",
        #         "us",
        #         "them",
        #         # German
        #         "der",
        #         "die",
        #         "das",
        #         "und",
        #         "oder",
        #         "aber",
        #         "in",
        #         "an",
        #         "zu",
        #         "für",
        #         "von",
        #         "mit",
        #         "bei",
        #         "über",
        #         "den",
        #         "ist",
        #         "sind",
        #         "war",
        #         "waren",
        #         "sein",
        #         "haben",
        #         "hat",
        #         "hatte",
        #         "werden",
        #         "wird",
        #         "wurde",
        #         # French
        #         "le",
        #         "la",
        #         "les",
        #         "et",
        #         "ou",
        #         "mais",
        #         "dans",
        #         "sur",
        #         "à",
        #         "pour",
        #         "de",
        #         "avec",
        #         "par",
        #         "est",
        #         "sont",
        #         "était",
        #         "étaient",
        #         "être",
        #         "avoir",
        #         "a",
        #         "avait",
        #         "sera",
        #         "serait",
        #     }
        #     self.nlp.Defaults.stop_words.update(common_stopwords)

        self.stop_words = stop_words if stop_words else set()
        self.nlp.Defaults.stop_words.update(self.stop_words)

        self.keep_num = keep_num
        self.keep_punct = keep_punct
        self.keep_space = keep_space
        self.keep_email = keep_email
        self.keep_url = keep_url
        self.keep_stopwords = keep_stopwords
        self.lower = lower

        if lang in ["multi", "xx"]:
            logger.info("Using multilingual model - supports mixed-language documents")
            logger.info("Note: Multilingual model has limited stopword support")

    def _add_mwe_patterns(self):
        """Add multi-word expression patterns to the spaCy pipeline"""
        if len(self.mwes) > 0 and "entity_ruler" not in self.nlp.pipe_names:
            # Add entity ruler before NER to ensure our patterns take precedence
            try:
                self.nlp.add_pipe("entity_ruler", before="ner")
            except ValueError:
                # If there's no NER component, just add it normally
                self.nlp.add_pipe("entity_ruler")

            ruler = self.nlp.get_pipe("entity_ruler")
            patterns = []
            for mwe in self.mwes:
                # Convert string MWE to token-based pattern for better multilingual support
                tokens = mwe.split()
                if len(tokens) > 1:
                    # Create pattern that matches the tokens (case-insensitive for multilingual)
                    pattern = [{"LOWER": token.lower()} for token in tokens]
                    patterns.append({"label": "MWE", "pattern": pattern})
                else:
                    # Single word - use string pattern
                    patterns.append({"label": "MWE", "pattern": mwe})

            if patterns:
                ruler.add_patterns(patterns)

    def is_valid_token(self, token):
        """Check if token should be kept based on filtering rules"""
        # Enhanced stopword checking for multilingual
        if not self.keep_stopwords:
            # Check both spaCy's is_stop and our custom stopword list
            if token.is_stop or token.text.lower() in self.nlp.Defaults.stop_words:
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

    def tokenize(self, text) -> list[str]:
        """
        Given a text, tokenize it into words and phrases.

        Args:
            text (str): The input text to tokenize (any supported language if lang='multi')

        Returns:
            list[str]: A list of tokens, including multi-word expressions.
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
    # Example 1: Multilingual usage with mixed-language documents
    print("=" * 60)
    print("Example 1: Multilingual MWE Detection")
    print("=" * 60)

    multilingual_texts = [
        "Machine learning is a subset of artificial intelligence.",
        "Das maschinelle Lernen ist ein Teilgebiet der künstlichen Intelligenz.",
        "L'apprentissage automatique est un sous-ensemble de l'intelligence artificielle.",
        "New York City is the largest city in the United States.",
        "Die Stadt New York ist die größte Stadt in den Vereinigten Staaten.",
    ]

    # Use 'multi' or 'xx' for multilingual support
    mwe_parser = MWEParser(
        lang="multi",  # This is the key change!
        min_count=2,
        threshold=0.5,
    )

    mwe_parser.learn_phraser(multilingual_texts)
    mwes = mwe_parser.extract_phrases()

    print(f"\nFound {len(mwes)} multi-word expressions across all languages")
    print(f"Sample MWEs: {mwes[:10]}")

    # Example 2: Tokenization with detected MWEs
    print("\n" + "=" * 60)
    print("Example 2: Multilingual Tokenization with MWEs")
    print("=" * 60)

    tokenizer = PhrasalTokenizer(
        lang="multi",  # Same here!
        mwes=mwes,
        concat_token="_",
        lower=True,
    )

    test_texts = [
        "Machine learning is used in New York City.",
        "Das maschinelle Lernen wird in New York verwendet.",
        "L'intelligence artificielle transforme notre société.",
    ]

    for text in test_texts:
        tokens = tokenizer.tokenize(text)
        print(f"\nText: {text}")
        print(f"Tokens: {tokens}")

    # Example 3: Single-language usage (unchanged API)
    print("\n" + "=" * 60)
    print("Example 3: Single Language Usage (Original API)")
    print("=" * 60)

    german_texts = [
        "Der Fotograf Jos Schmit hat für Dies academicus gearbeitet.",
        "Jos Schmit und Maria Müller arbeiten für Dies academicus.",
    ]

    # Original API still works exactly the same
    single_lang_parser = MWEParser(
        lang="de",
        connector_words=["und", "oder", "für"],
    )

    single_lang_parser.learn_phraser(german_texts)
    german_mwes = single_lang_parser.extract_phrases()

    print(f"\nFound {len(german_mwes)} German MWEs")
    print(f"German MWEs: {german_mwes}")

    single_lang_tokenizer = PhrasalTokenizer(
        lang="de",
        mwes=german_mwes,
        concat_token="_",
        lower=False,
    )

    tokens = single_lang_tokenizer.tokenize(german_texts[0])
    print(f"\nTokens: {tokens}")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("To use multilingual mode: simply set lang='multi'")
    print("To use single language: set lang='en', 'de', 'fr', etc.")

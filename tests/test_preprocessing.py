def test_phrasal_tokenizer():
    from text_tools.preprocessing import PhrasalTokenizer

    tokenizer = PhrasalTokenizer(
        lang="de", mwes=["Dies academicus"], concat_token="_", lower=False
    )

    sentence = "Der Fotograf Jos Schmit hat für Dies academicus gearbeitet."
    tokenized = tokenizer.tokenize(sentence)

    assert tokenized == ["Fotograf", "Jos_Schmit", "Dies_academicus", "gearbeitet"]


def test_mwe_parser():
    from text_tools.preprocessing import MWEParser
    from text_tools.data import ChunkedDataset
    from text_tools.constants import CONNECTOR_WORDS

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

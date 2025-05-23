def test_build_chunked_dataset():
    from datasets import Dataset
    from text_tools.data import ChunkedDataset

    input_dir = "tests/data"

    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    max_tokens = 32

    dataset = ChunkedDataset(
        model_id=model_id,
        max_tokens=max_tokens,
    )

    dataset.build_chunked_dataset(
        input_dir=input_dir,
        extensions=[".md"],
        recursive=True,
    )

    # check that the resulting dataset is a Dataset object
    assert isinstance(dataset.dataset, Dataset)

    # check that the dataset has the expected number of columns
    assert len(dataset.dataset.column_names) == 2

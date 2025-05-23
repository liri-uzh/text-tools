import pytest
from pathlib import Path
import shutil
from text_tools.data import ChunkedDataset # Assuming your class is in a file named 'data.py'

# Define a temporary directory for tests
@pytest.fixture(scope="module")
def temp_test_dir(tmp_path_factory):
    """Creates a temporary directory for tests."""
    temp_dir = tmp_path_factory.mktemp("chunked_dataset_tests")
    yield temp_dir
    # Clean up after tests
    shutil.rmtree(temp_dir)

@pytest.fixture(scope="module")
def setup_dummy_data(temp_test_dir):
    """Sets up dummy markdown files for testing."""
    data_dir = temp_test_dir / "test_data"
    data_dir.mkdir()

    (data_dir / "file1.md").write_text("# Title 1\nThis is some content for file 1.")
    (data_dir / "file2.md").write_text("## Subtitle 2\nAnother file with more text here. This file has more tokens.")
    (data_dir / "file3.txt").write_text("This is a text file that should be ignored.")

    nested_dir = data_dir / "nested"
    nested_dir.mkdir()
    (nested_dir / "nested_file.md").write_text("### Nested Title\nContent from a nested markdown file.")

    return data_dir

@pytest.fixture(scope="module")
def output_dir(temp_test_dir):
    """Creates a temporary output directory for saving datasets."""
    out_dir = temp_test_dir / "test_output"
    out_dir.mkdir()
    return out_dir

def test_build_chunked_dataset_non_recursive(setup_dummy_data):
    """Tests building a dataset non-recursively with markdown files."""
    input_dir = setup_dummy_data
    dataset = ChunkedDataset(
        input_dir=str(input_dir),
        extensions=[".md"],
        recursive=False,
        model_id="sentence-transformers/all-MiniLM-L6-v2"
    )
    assert dataset is not None
    
    assert len(dataset) > 0
    # Should only contain files from the top-level directory
    assert any("file1" in d["file"] for d in dataset)
    assert any("file2" in d["file"] for d in dataset)
    assert not any("nested_file" in d["file"] for d in dataset)
    assert hasattr(dataset, 'tokenizer') # Ensure tokenizer is attached

def test_build_chunked_dataset_recursive(setup_dummy_data):
    """Tests building a dataset recursively with markdown files."""
    input_dir = setup_dummy_data
    dataset = ChunkedDataset(
        input_dir=str(input_dir),
        extensions=[".md"],
        recursive=True,
        model_id="sentence-transformers/all-MiniLM-L6-v2"
    )
    assert dataset is not None
    assert len(dataset) > 0
    assert any("file1" in d["file"] for d in dataset)
    assert any("file2" in d["file"] for d in dataset)
    assert any("nested_file" in d["file"] for d in dataset)
    assert hasattr(dataset, 'tokenizer')

def test_build_chunked_dataset_invalid_dir():
    """Tests building a dataset from an invalid directory."""
    with pytest.raises(ValueError, match="Directory .* does not exist."):
        ChunkedDataset(
            input_dir="/non/existent/path/to/data",
            extensions=[".md"],
            recursive=False,
            model_id="sentence-transformers/all-MiniLM-L6-v2"
        )

def test_build_chunked_dataset_no_files_found(temp_test_dir):
    """Tests building a dataset when no matching files are found."""
    empty_dir = temp_test_dir / "empty_data"
    empty_dir.mkdir()
    (empty_dir / "other.txt").write_text("some content") # Not a .md file

    with pytest.raises(ValueError, match="No files found in .* with extensions .*"):
        ChunkedDataset(
            input_dir=str(empty_dir),
            extensions=[".md"],
            recursive=False,
            model_id="sentence-transformers/all-MiniLM-L6-v2"
        )

def test_save_and_load_chunked_dataset(setup_dummy_data, output_dir):
    """Tests saving and loading a chunked dataset."""
    input_dir = setup_dummy_data
    save_path = output_dir / "saved_dataset"

    # 1. Build and save
    dataset = ChunkedDataset(
        input_dir=str(input_dir),
        extensions=[".md"],
        recursive=False,
        model_id="sentence-transformers/all-MiniLM-L6-v2"
    )
    ChunkedDataset.save_chunked_dataset(dataset, str(save_path))
    
    assert save_path.exists()
    assert (save_path / "dataset_info.json").exists() # Hugging Face dataset file
    assert (save_path / "tokenizer_config.json").exists() # Tokenizer file

    # 2. Load the dataset
    loaded_dataset = ChunkedDataset(load_from_path=str(save_path))
    assert loaded_dataset is not None
    assert len(loaded_dataset) == len(dataset)
    assert loaded_dataset.column_names == dataset.column_names
    assert hasattr(loaded_dataset, 'tokenizer') # Ensure tokenizer is loaded and attached
    
    # Check content of loaded dataset
    original_texts = sorted([d["text"] for d in dataset])
    loaded_texts = sorted([d["text"] for d in loaded_dataset])
    assert original_texts == loaded_texts

def test_load_chunked_dataset_invalid_path():
    """Tests loading a dataset from a non-existent path."""
    with pytest.raises(ValueError, match="Directory .* does not exist."):
        ChunkedDataset(load_from_path="/non/existent/saved/dataset")

def test_add_length_column(setup_dummy_data):
    """Tests adding the tokenized length column to the dataset."""
    input_dir = setup_dummy_data
    dataset = ChunkedDataset(
        input_dir=str(input_dir),
        extensions=[".md"],
        recursive=False,
        model_id="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Add length column
    dataset_with_length = ChunkedDataset.add_length_column(dataset, n_processes=1) # Use 1 process for consistent testing

    assert "token_count" in dataset_with_length.column_names
    assert len(dataset_with_length["token_count"]) == len(dataset_with_length)
    assert all(isinstance(count, int) for count in dataset_with_length["token_count"])
    assert all(count > 0 for count in dataset_with_length["token_count"])
    
    
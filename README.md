# `text-tools`

A collection of tools for processing large text corpora (e.g. directories of markdown files) and for tokenizing text with multi-word expressions (MWEs).

---

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
  - [Standard Installation](#standard-installation)
  - [Development Installation](#development-installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

---

## Introduction

`text-tools` is a Python library designed to provide a set of reusable and efficient tools for common text processing tasks. Whether you're dealing with natural language processing, data cleaning, or general text manipulation, this library aims to streamline your workflow.

---

## Features

- **ChunkedDataset**: Given a directory of corpus files in markdown format, this class can load the files and create a dataset of text chunks, which can be used for various text processing tasks.
- **PhrasalTokenizer**: A tokenizer that recognizes pre-computed multi-word expressions (MWEs) and treats them as single tokens during tokenization.

---

## Installation

To install the `text-tools` library, we recommend installing from source.

```bash
git clone [https://github.com/your-username/text-tools.git](https://github.com/your-username/text-tools.git)
cd text-tools
pip install .

# if you plan to contribute to the project or want to run tests, you can install the development dependencies:
pip install -e ".[dev]"
```

## Usage

### ChunkedDataset

Here's a quick example of how to use the text chunking module:

```python
dataset = ChunkedDataset(
    input_dir="path/to/directory/containing/markdown/files",
    extensions=[".md"],
    recursive=True,
    model_id="sentence-transformers/all-MiniLM-L6-v2"
    max_tokens=256, # should be set to the model's max input length
    )

# save the chunked dataset to disk
ChunkedDataset.save_chunked_dataset(dataset, output_dir="tests/data/chunked_data")

# load the chunked dataset from disk
dataset = ChunkedDataset(load_from_path="tests/data/chunked_data")
```

### PhrasalTokenizer

Here's a quick example of how to use the phrasal tokenizer:

```python

from stopwordsiso import stopwords
from text_tools.data import ChunkedDataset

lang = "de"

mwe_parser = MWEParser(
    lang=lang,
    connector_words=stopwords([lang]),
)

# load a chunked dataset
dataset = ChunkedDataset.load_chunked_dataset(input_dir="tests/data/chunked_data")

# learn multi-word expressions from the dataset
mwe_parser.learn_phraser(dataset["text"])
mwes = mwe_parser.extract_phrases()

# initialise a PhrasalTokenizer with the learned multi-word expressions
tokenizer = PhrasalTokenizer(
    lang=lang,
    mwes=mwes,
    concat_token="_",
    stop_words=stopwords([lang]),
    keep_stopwords=False,
    lower=False
    )

# tokenize a sentence
tokenizer.tokenize("Der Fotograf Jos Schmit hat für Dies academicus gearbeitet.")
>>> ['Fotograf', 'Jos_Schmit', 'Dies_academicus', 'gearbeitet']
```

## Contributing

We welcome contributions to text-tools! If you have suggestions for improvements, new features, or bug fixes, please open an issue or submit a pull request.

To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Make your changes.
4. Write and run tests to ensure everything works as expected.
5. Commit your changes (`git commit -m 'Add new feature'`).
6. Push to your branch (`git push origin feature/your-feature`).
7. Open a pull request.

Please run `ruff check` and `ruff format` to check and format your code before submitting a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

`text-tools` uses the following libraries:

- `transformers`: For pre-trained models and tokenization.
- `datasets`: For dataset handling and processing.
- `docling`: For markdown document processing and chunking.
- `gensim`: For learning multi-word expressions
- `spacy`: For tokenization and linguistic features.

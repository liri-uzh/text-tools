# /usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List, Generator, Union
from pathlib import Path
from tqdm import tqdm
import logging

from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.serializer.markdown import MarkdownTableSerializer

# set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MDTableSerializerProvider(ChunkingSerializerProvider):
    """
    Serializer provider for Markdown tables.
    Taken from docling documentation (https://docling-project.github.io/docling/examples/advanced_chunking_and_serialization/#table-serialization)
    """

    def get_serializer(self, doc):
        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=MarkdownTableSerializer(),
        )


class ChunkedDataset:
    """
    Class to process a collection of long text documents into a Dataset containing smaller chunks.
    """

    def __init__(
        self,
        model_id: str = None,
        max_tokens: int = 128,
        merge_peers: bool = True,
        serialize_tables: bool = True,
    ) -> None:
        """
        Initialize the ChunkingProcessor.

        Args:

            model_id (str): Model ID for the tokenizer.

            max_tokens (int): Maximum number of tokens per chunk.

            merge_peers (bool): Whether to merge peer chunks (see docling documentation).

            serialize_tables (bool): Whether to serialize tables (see https://docling-project.github.io/docling/examples/advanced_chunking_and_serialization/#configuring-a-different-strategy)
        """

        self.model_id = model_id
        self.max_tokens = max_tokens
        self.merge_peers = merge_peers
        self.serialize_tables = serialize_tables

        self.tokenizer = AutoTokenizer.from_pretrained(model_id) if model_id else None
        self.dataset = None

    def build_chunked_dataset(
        self,
        input_dir: str,
        extensions: List[str] = [".md"],
        recursive: bool = True,
    ) -> Dataset:
        """
        Given a directory, we load all files with the specified extensions
        and create a chunked dataset from them.

        Args:
            input_dir (str): Directory containing the files to chunk.
            extensions (List[str]): List of file extensions to include.
            recursive (bool): Whether to search recursively in subdirectories.

        Returns:
            Dataset: A dataset of chunks.
        """

        if not Path(input_dir).exists():
            raise ValueError(f"Directory {input_dir} does not exist.")

        files = []

        if recursive:
            logger.info(
                f"Searching for files with extensions {extensions} in {input_dir} recursively..."
            )
            glob_pattern = r"**/*"
        else:
            logger.info(
                f"Searching for files with extensions {extensions} in {input_dir}..."
            )
            glob_pattern = r"*"
        for path in Path(input_dir).glob(glob_pattern):
            logger.debug(f"Found file: {path}")
            if path.is_file and path.suffix in extensions:
                files.append(path)

        if len(files) == 0:
            raise ValueError(
                f"No files found in {input_dir} with extensions {extensions}."
            )

        logger.info(f"Building chunked dataset from {len(files)} files...")

        chunks = list(self.chunked_dataset_generator(files=files))

        self.dataset = Dataset.from_list(chunks)

        return self.dataset

    def chunked_dataset_generator(
        self,
        files: Union[List[str], Generator[str, None, None]],
    ) -> Generator[dict, None, None]:
        """
        Process a collection of long text documents yielding smaller chunks.

        Args:
            files (List[str]): List of files to chunk.
        """

        chunker = HybridChunker(
            tokenizer=self.tokenizer,  # instance or model name, defaults to "sentence-transformers/all-MiniLM-L6-v2"
            max_tokens=self.max_tokens,  # optional, by default derived from `tokenizer`
            merge_peers=self.merge_peers,  # optional, defaults to True
            serializer_provider=MDTableSerializerProvider()
            if self.serialize_tables
            else None,
        )

        for file in tqdm(files, desc="Chunking files", unit="file"):
            doc = DocumentConverter().convert(source=file).document
            for chunk in chunker.chunk(dl_doc=doc):
                yield {
                    "text": chunk.text,
                    "file": file.stem,
                }

    def save_chunked_dataset(
        self,
        output_dir: str,
    ) -> None:
        """
        Save the chunked dataset to a directory.

        Args:
            dataset (Dataset): The chunked dataset to save.
            output_dir (str): Directory to save the dataset.
        """

        if not Path(output_dir).exists():
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            logger.info(f"Creating directory {output_dir}...")
        else:
            logger.info(f"Directory {output_dir} already exists. Overwriting...")

        self.dataset.save_to_disk(output_dir)
        logger.info(f"Chunked dataset saved to {output_dir}")
        # save the tokenizer
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Tokenizer saved to {output_dir}")

        return

    def load_chunked_dataset(
        self,
        data_dir: str,
    ):
        """
        Load a chunked dataset from a directory.

        Args:
            data_dir (str): Directory containing the chunked dataset
            and the corresponding tokenizer.

        Returns:
            Dataset: The loaded chunked dataset.
        """
        if not Path(data_dir).exists():
            raise ValueError(f"Directory {data_dir} does not exist.")

        self.dataset = load_from_disk(data_dir)
        logger.info(f"Chunked dataset loaded from {data_dir}")
        # load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(data_dir)
        logger.info(f"Tokenizer loaded from {data_dir}")

    @staticmethod
    def get_tokenized_length(batch, tokenizer):
        # Tokenize the batch of texts without padding
        tokenized = tokenizer(batch["text"], truncation=False, padding=False)
        # Extract the tokenized lengths for each text in the batch
        token_count = [len(tokens) for tokens in tokenized["input_ids"]]
        # Return a dictionary with the token lengths for each item
        return {"token_count": token_count}

    def add_length_column(
        self,
        n_processes: int = 4,
    ) -> None:
        """
        Add a column to the dataset with the tokenized length of each chunk.

        Args:
            n_processes (int): Number of processes to use for tokenization.
        """

        if not hasattr(self, "dataset"):
            raise ValueError("Dataset not loaded. Please load the dataset first.")

        self.dataset = self.dataset.map(
            self.get_tokenized_length,
            fn_kwargs={"tokenizer": self.tokenizer},
            batched=True,
            num_proc=n_processes,
        )

        logger.info("Tokenized length column added to dataset.")


if __name__ == "__main__":
    pass

    # Example usage:

    # Construct a chunked dataset from a directory of files
    # dataset = ChunkedDataset(model_id="sentence-transformers/all-MiniLM-L6-v2")
    # dataset.build_chunked_dataset(input_dir="tests/data", extensions=[".md"], recursive=True)
    # dataset.save_chunked_dataset(output_dir="tests/chunked_data")

    # Load a previously saved dataset
    # dataset = ChunkedDataset()
    # dataset.load_chunked_dataset(data_dir="tests/chunked_data")
    # dataset.add_length_column(n_processes=4)
    # print(dataset.dataset)

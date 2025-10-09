# /usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List
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


class ChunkedDataset(Dataset):
    """
    Class to process a collection of long text documents into a Dataset containing smaller chunks.
    """

    def __new__(
        cls,
        input_dir: str = None,
        extensions: List[str] = [".md"],
        recursive: bool = True,
        model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_tokens: int = 128,
        merge_peers: bool = True,
        serialize_tables: bool = True,
        load_from_path: str = None,
    ) -> Dataset:
        """
        Initialize and build/load the chunked dataset. This acts as the constructor
        for the Dataset object itself.

        Args:
            input_dir (str): Directory containing the files to chunk.
            extensions (List[str]): List of file extensions to include.
            recursive (bool): Whether to search recursively in subdirectories.
            model_id (str): Model ID for the tokenizer.
            max_tokens (int): Maximum number of tokens per chunk.
            merge_peers (bool): Whether to merge peer chunks (see docling documentation).
            serialize_tables (bool): Whether to serialize tables.
            load_from_path (str): Path to load a previously saved dataset from.
                                  If provided, other parameters for building are ignored.

        Returns:
            Dataset: A dataset of chunks.
        """
        if load_from_path:
            if not Path(load_from_path).exists():
                raise ValueError(f"Directory {load_from_path} does not exist.")
            logger.info(f"Loading chunked dataset from {load_from_path}")
            dataset_obj = load_from_disk(load_from_path)
            # Load the tokenizer separately as it's not part of the Dataset object itself
            dataset_obj.tokenizer = AutoTokenizer.from_pretrained(load_from_path)
            logger.info(f"Tokenizer loaded from {load_from_path}")
            return dataset_obj

        if not input_dir:
            raise ValueError("Either 'input_dir' or 'load_from_path' must be provided.")

        tokenizer = AutoTokenizer.from_pretrained(model_id)

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
            if path.is_file() and path.suffix in extensions:
                files.append(path)

        if len(files) == 0:
            raise ValueError(
                f"No files found in {input_dir} with extensions {extensions}."
            )

        logger.info(f"Building chunked dataset from {len(files)} files...")

        chunker = HybridChunker(
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            merge_peers=merge_peers,
            serializer_provider=MDTableSerializerProvider()
            if serialize_tables
            else None,
        )

        chunks_data = []
        for file in tqdm(files, desc="Chunking files", unit="file"):
            doc = DocumentConverter().convert(source=file).document
            for chunk in chunker.chunk(dl_doc=doc):
                chunks_data.append(
                    {
                        "text": chunk.text,
                        "file": file.stem,
                    }
                )

        dataset_obj = Dataset.from_list(chunks_data)
        # Attach the tokenizer to the dataset object for later use
        dataset_obj.tokenizer = tokenizer
        return dataset_obj

    @staticmethod
    def save_chunked_dataset(
        dataset: Dataset,
        output_dir: str,
    ) -> None:
        """
        Save the chunked dataset to a directory.
        """
        if not Path(output_dir).exists():
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            logger.info(f"Creating directory {output_dir}...")
        else:
            logger.info(f"Directory {output_dir} already exists. Overwriting...")

        dataset.save_to_disk(output_dir)
        logger.info(f"Chunked dataset saved to {output_dir}")
        # save the tokenizer associated with the dataset
        if hasattr(dataset, "tokenizer") and dataset.tokenizer:
            dataset.tokenizer.save_pretrained(output_dir)
            logger.info(f"Tokenizer saved to {output_dir}")
        else:
            logger.warning(
                "Tokenizer not found on the dataset object. Skipping tokenizer save."
            )

        return

    @staticmethod
    def save_chunked_dataset_as_jsonl(
        dataset: Dataset,
        output_file: str,
    ) -> None:
        """
        Save the chunked dataset to a JSONL file.
        """
        dataset.to_json(output_file, force_ascii=False)
        
        logger.info(f"Chunked dataset saved to {output_file} in JSONL format.")
        
        return

    @staticmethod
    def get_tokenized_length(batch, tokenizer):
        # Tokenize the batch of texts without padding
        tokenized = tokenizer(batch["text"], truncation=False, padding=False)
        # Extract the tokenized lengths for each text in the batch
        token_count = [len(tokens) for tokens in tokenized["input_ids"]]
        # Return a dictionary with the token lengths for each item
        return {"token_count": token_count}

    @staticmethod
    def add_length_column(
        dataset: Dataset,
        n_processes: int = 4,
    ) -> Dataset:
        """
        Add a column to the dataset with the tokenized length of each chunk.
        """
        if not hasattr(dataset, "tokenizer") or not dataset.tokenizer:
            raise ValueError(
                "Tokenizer not found on the dataset. Please ensure it was loaded or set."
            )

        dataset = dataset.map(
            ChunkedDataset.get_tokenized_length,  # Call the static method
            fn_kwargs={"tokenizer": dataset.tokenizer},
            batched=True,
            num_proc=n_processes,
        )
        logger.info("Tokenized length column added to dataset.")
        return dataset


if __name__ == "__main__":
    # Example usage:

    # Construct a chunked dataset from a directory of files
    # Make sure 'tests/data' exists and contains markdown files
    # You would typically pass the input_dir here
    dataset = ChunkedDataset(
        input_dir="tests/data",
        extensions=[".md"],
        recursive=False,
        model_id="sentence-transformers/all-MiniLM-L6-v2",
    )
    print(dataset)  # This directly prints the Dataset object

    # To save:
    ChunkedDataset.save_chunked_dataset(dataset, output_dir="tests/chunked_data")

    # Load a previously saved dataset
    # dataset = ChunkedDataset(load_from_path="tests/chunked_data")
    # print(dataset)

    # Add length column to the loaded dataset
    # dataset = ChunkedDataset.add_length_column(dataset, n_processes=4)
    # print(dataset)
    # print(dataset[0]['token_count'])

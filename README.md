# Advanced Text Segmentation

This repository contains a state-of-the-art, open-source Python package for advanced text segmentation (chunking). It is designed to be a critical component in Retrieval-Augmented Generation (RAG) systems and advanced Natural Language Processing (NLP) data pipelines.

The package transforms large, unstructured or semi-structured documents into optimally sized segments (chunks) that maximize semantic coherence while adhering to token constraints.

## Core Features

- **Multiple Chunking Strategies**: From simple fixed-size and recursive splitting to advanced, structure-aware, and semantic-based strategies.
- **Rich Metadata**: Every chunk is enriched with detailed metadata, including its start and end position in the original document, hierarchical context, and more.
- **Framework Integration**: Seamless integration with popular RAG frameworks like LangChain and LlamaIndex.
- **Highly Configurable**: All strategies are hyper-parameterized with sensible, research-backed defaults.
- **Extensible Architecture**: The modular design makes it easy to implement, customize, and combine strategies.

## Installation

You can install the core package and its dependencies using pip. The package is structured with optional extras for strategies that require heavy dependencies.

```bash
# Install the core package
pip install .

# To include support for sentence splitting (via NLTK)
pip install .[nlp]

# To include support for Markdown and HTML splitting
pip install .[markdown,html]

# To include support for semantic splitting (requires numpy)
pip install .[semantic]

# To include support for code splitting (requires tree-sitter)
pip install .[code]

# To install framework integrations
pip install .[langchain,llamaindex]

# To install everything for development
pip install .[dev,nlp,markdown,html,semantic,code,langchain,llamaindex]
```

## Global Configuration

All splitter classes inherit from a common `TextSplitter` base class and share a set of powerful configuration options for preprocessing and chunk management.

- `normalize_whitespace` (bool, default: `False`): If `True`, collapses all consecutive whitespace characters (spaces, newlines, tabs) into a single space. This is useful for cleaning up messy text but may affect index mapping for some strategies.
- `unicode_normalize` (str, default: `None`): Specifies a Unicode normalization form to apply (e.g., `'NFC'`, `'NFKC'`). Helps ensure consistent character representation.
- `minimum_chunk_size` (int, default: `0`): If set, the splitter will attempt to handle chunks smaller than this size.
- `min_chunk_merge_strategy` (str, default: `'merge_with_previous'`): Defines how to handle small chunks.
    - `'merge_with_previous'`: Merges a small chunk with the one that came before it. If the first chunk is too small, it is kept as is.
    - `'discard'`: Simply removes any chunk smaller than `minimum_chunk_size`.

## Quick Start

The simplest way to get started is with the `RecursiveCharacterSplitter`, which is a good general-purpose splitter.

```python
from text_segmentation.strategies.recursive import RecursiveCharacterSplitter

text = "This is a long document.   It has multiple sentences and paragraphs.\n\nWe want to split it into smaller chunks. Some chunks are small."

# Initialize the splitter with global options
splitter = RecursiveCharacterSplitter(
    chunk_size=50,
    chunk_overlap=10,
    normalize_whitespace=True,
    minimum_chunk_size=20,
    min_chunk_merge_strategy="discard"
)

# Split the text
chunks = splitter.split_text(text)

for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} ---")
    print(f"Content: {chunk.content}")
    print(f"Start Index: {chunk.start_index}")
    print(f"Metadata: {chunk.to_dict()}")
```

## Strategies

### FixedSizeSplitter
The most basic strategy. Splits text into chunks of a fixed character size.

```python
from text_segmentation.strategies.fixed_size import FixedSizeSplitter
splitter = FixedSizeSplitter(chunk_size=100, chunk_overlap=20)
chunks = splitter.split_text(my_text)
```

### RecursiveCharacterSplitter
Recursively splits text based on a prioritized list of separators (e.g., `["\n\n", "\n", ". ", " "]`). This is often the recommended starting point.

```python
from text_segmentation.strategies.recursive import RecursiveCharacterSplitter
splitter = RecursiveCharacterSplitter(chunk_size=1024, chunk_overlap=200)
chunks = splitter.split_text(my_text)
```

### SentenceSplitter
Splits text based on sentence boundaries, then aggregates sentences into chunks. Requires the `[nlp]` extra.

```python
from text_segmentation.strategies.sentence import SentenceSplitter
# Ensure you have run: python -c "import nltk; nltk.download('punkt')"
splitter = SentenceSplitter(chunk_size=1024, overlap_sentences=1)
chunks = splitter.split_text(my_prose_text)
```

### MarkdownSplitter
A structure-aware splitter that uses Markdown headers (H1-H6), paragraphs, and other elements as boundaries. Requires the `[markdown]` extra.

```python
from text_segmentation.strategies.structure.markdown import MarkdownSplitter
splitter = MarkdownSplitter(chunk_size=1024, chunk_overlap=0)
chunks = splitter.split_text(my_markdown_text)
# Chunks will have `hierarchical_context` metadata populated.
print(chunks[0].hierarchical_context)
# Output: {'H1': 'Main Title', 'H2': 'Section 1'}
```

### HTMLSplitter
A structure-aware splitter for HTML documents. Requires the `[html]` extra.

```python
from text_segmentation.strategies.structure.html import HTMLSplitter
splitter = HTMLSplitter(chunk_size=1024, chunk_overlap=0)
chunks = splitter.split_text(my_html_text)
```

### SemanticSplitter
Splits text by finding semantic breakpoints between sentences using an embedding model. Requires the `[semantic]` extra.

```python
from text_segmentation.strategies.semantic import SemanticSplitter
# You must provide your own embedding function.
# e.g., from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('all-MiniLM-L6-v2')
# embedding_function = model.encode

splitter = SemanticSplitter(
    embedding_function=my_embedding_function,
    breakpoint_method="percentile",
    breakpoint_threshold=95
)
chunks = splitter.split_text(my_text)
```

### CodeSplitter
A syntax-aware splitter for source code. Requires the `[code]` extra.

```python
from text_segmentation.strategies.code import CodeSplitter
splitter = CodeSplitter(language="python", chunk_size=1024, chunk_overlap=0)
chunks = splitter.split_text(my_python_code)
```

## Framework Integrations

### LangChain
Use any splitter in a LangChain pipeline. Requires the `[langchain]` extra.
```python
from text_segmentation.strategies.recursive import RecursiveCharacterSplitter
from text_segmentation.integrations.langchain import LangChainWrapper

# 1. Create a splitter instance from this package
ats_splitter = RecursiveCharacterSplitter(chunk_size=100, chunk_overlap=10)

# 2. Wrap it for LangChain
langchain_splitter = LangChainWrapper(ats_splitter)

# 3. Use it like any other LangChain TextSplitter
from langchain_core.documents import Document
docs = [Document(page_content="Some long text...")]
split_docs = langchain_splitter.split_documents(docs)
print(split_docs[0].metadata)
```

### LlamaIndex
Use any splitter as a LlamaIndex `NodeParser`. Requires the `[llamaindex]` extra.
```python
from text_segmentation.strategies.sentence import SentenceSplitter
from text_segmentation.integrations.llamaindex import LlamaIndexWrapper

# 1. Create a splitter instance
ats_splitter = SentenceSplitter(chunk_size=512, overlap_sentences=1)

# 2. Create the LlamaIndex-compatible NodeParser
node_parser = LlamaIndexWrapper(ats_splitter)

# 3. Use it in your LlamaIndex pipeline
from llama_index.core.schema import Document
nodes = node_parser.get_nodes_from_documents([Document(text="Some long text...")])
print(nodes[0].metadata)
```

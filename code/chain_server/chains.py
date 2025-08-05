# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LLM Chains for executing Retrieval Augmented Generation."""
import base64
import os
import time
import json
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Generator, List, Optional

import requests
import torch
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from llama_index.embeddings import LangchainEmbedding
from llama_index import (
    Prompt,
    ServiceContext,
    VectorStoreIndex,
    download_loader,
    set_global_service_context,
)
from llama_index.postprocessor.types import BaseNodePostprocessor
from llama_index.node_parser import SimpleNodeParser
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.response.schema import StreamingResponse, Response
from llama_index.schema import MetadataMode
from llama_index.utils import globals_helper, get_tokenizer
from llama_index.vector_stores import MilvusVectorStore, SimpleVectorStore

from chain_server import configuration, chat_templates

if TYPE_CHECKING:
    from llama_index.indices.base_retriever import BaseRetriever
    from llama_index.indices.query.schema import QueryBundle
    from llama_index.schema import NodeWithScore
    from llama_index.types import TokenGen
    from chain_server.configuration_wizard import ConfigWizard

TEXT_SPLITTER_MODEL = "intfloat/e5-large-v2"
TEXT_SPLITTER_CHUNCK_SIZE = 510
TEXT_SPLITTER_CHUNCK_OVERLAP = 200
EMBEDDING_MODEL = "intfloat/e5-large-v2"
DEFAULT_NUM_TOKENS = 50
DEFAULT_MAX_CONTEXT = 800


class LimitRetrievedNodesLength(BaseNodePostprocessor):
    """Llama Index chain filter to limit token lengths."""

    def _postprocess_nodes(
        self, nodes: List["NodeWithScore"] = [], query_bundle: Optional["QueryBundle"] = None
    ) -> List["NodeWithScore"]:
        included_nodes = []
        current_length = 0
        limit = DEFAULT_MAX_CONTEXT
        tokenizer = get_tokenizer()

        for node in nodes:
            current_length += len(
                tokenizer(
                    node.get_content(metadata_mode=MetadataMode.LLM)
                )
            )
            if current_length > limit:
                break
            included_nodes.append(node)

        return included_nodes


@lru_cache
def get_config() -> "ConfigWizard":
    config_file = os.environ.get("APP_CONFIG_FILE", "/dev/null")
    config = configuration.AppConfig.from_file(config_file)
    if config:
        return config
    raise RuntimeError("Unable to find configuration.")


@lru_cache
def stream_ollama(prompt: str,
                  num_tokens: int,
                  temp: float,
                  top_p: float,
                  freq_pen: float,
                  pres_pen: float) -> Generator[str, None, None]:
    """Stream a completion from a local Ollama server."""
    url = "http://localhost:11434/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "llama3.1:8b-instruct-q8_0",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": num_tokens,
        "temperature": temp,
        "top_p": top_p,
        "frequency_penalty": freq_pen,
        "presence_penalty": pres_pen,
        "stream": True,
    }
    with requests.post(url, headers=headers, json=data, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            decoded = line.decode("utf-8")
            if decoded.strip() == "data: [DONE]":
                break
            if decoded.startswith("data: "):
                payload = json.loads(decoded[6:])
                content = payload["choices"][0]["delta"].get("content")
                if content:
                    yield content


@lru_cache
def get_embedding_model() -> LangchainEmbedding:
    model_kwargs = {"device": "cpu"}
    device_str = os.environ.get('EMBEDDING_DEVICE', "cpu")
    if torch.cuda.is_available():
        model_kwargs["device"] = device_str

    encode_kwargs = {"normalize_embeddings": False}
    hf_embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    return LangchainEmbedding(hf_embeddings)


@lru_cache
def get_vector_index() -> VectorStoreIndex:
    config = get_config()
    vector_store = MilvusVectorStore(uri=config.milvus, dim=1024, overwrite=False)
    return VectorStoreIndex.from_vector_store(vector_store)


@lru_cache
def get_doc_retriever(num_nodes: int = 4) -> "BaseRetriever":
    index = get_vector_index()
    return index.as_retriever(similarity_top_k=num_nodes)


@lru_cache
def set_service_context() -> None:
    service_context = ServiceContext.from_defaults(
        embed_model=get_embedding_model()
    )
    set_global_service_context(service_context)


def llm_chain_streaming(
    context: str,
    question: str,
    num_tokens: int,
    temp: float,
    top_p: float,
    freq_pen: float,
    pres_pen: float,
) -> Generator[str, None, None]:
    """Execute a simple LLM chain using the components defined above."""
    set_service_context()
    prompt = chat_templates.LLAMA_3_CHAT_TEMPLATE.format(context_str=context, query_str=question)
    start = time.time()
    response = stream_ollama(prompt, num_tokens, temp, top_p, freq_pen, pres_pen)
    perf = time.time() - start
    yield str(perf * 1000).split('.', 1)[0]
    for chunk in response:
        yield chunk


def rag_chain_streaming(prompt: str,
                        num_tokens: int,
                        temp: float,
                        top_p: float,
                        freq_pen: float,
                        pres_pen: float) -> Generator[str, None, None]:
    """Execute a Retrieval Augmented Generation chain using the components defined above."""
    set_service_context()
    nodes = get_doc_retriever(num_nodes=2).retrieve(prompt)
    docs = [node.get_text() for node in nodes]
    prompt = chat_templates.LLAMA_3_RAG_TEMPLATE.format(context_str=", ".join(docs), query_str=prompt)
    start = time.time()
    completion = stream_ollama(prompt, num_tokens, temp, top_p, freq_pen, pres_pen)
    perf = time.time() - start
    yield str(perf * 1000).split('.', 1)[0]
    for chunk in completion:
        yield chunk


def is_base64_encoded(s: str) -> bool:
    try:
        decoded_bytes = base64.b64decode(s)
        decoded_str = decoded_bytes.decode("utf-8")
        return s == base64.b64encode(decoded_str.encode("utf-8")).decode("utf-8")
    except Exception:  # pylint:disable = broad-exception-caught
        return False


def ingest_docs(data_dir: str, filename: str) -> None:
    unstruct_reader = download_loader("UnstructuredReader")
    loader = unstruct_reader()
    documents = loader.load_data(file=Path(data_dir), split_documents=False)

    encoded_filename = filename[:-4]
    if not is_base64_encoded(encoded_filename):
        encoded_filename = base64.b64encode(encoded_filename.encode("utf-8")).decode("utf-8")

    for document in documents:
        document.metadata = {"filename": encoded_filename}

    index = get_vector_index()
    node_parser = SimpleNodeParser.from_defaults()
    nodes = node_parser.get_nodes_from_documents(documents)
    index.insert_nodes(nodes)

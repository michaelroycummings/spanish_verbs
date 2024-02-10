from typing import List, Dict, Tuple,  Generator
import requests
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch


def iter_corpus(batch_size: int = 100, offset: int = 0) -> Generator:
    """
    Returns an iterable that retrieves a list of sentences
    from the wikipedia's spanish dataset on HuggingFace.
        Once the end of the dataset is reached, it will return an empty list []
        for every subsequent call.
    """
    dataset = 'wikimedia/wikipedia'
    config_name = '20231101.es'
    split = 'train'

    base_url = "https://datasets-server.huggingface.co/rows"

    batch_size = min(batch_size, 100) # Max batch_size is 100

    while True:
        params = {
            "dataset": dataset,
            "config": config_name,
            "split": split,
            "offset": offset,
            "length": batch_size
        }
        try_count = 0
        try:
            try_count += 1
            response = requests.get(base_url, params=params, timeout=5)
            data = response.json()
            results = [instance["row"]["text"] for instance in data["rows"]]
        except Exception:  # pylint: disable=broad-except
            if try_count >= 3:
                raise
            continue
        yield results

        offset += batch_size

def map_words_to_tokens(
    word_range: Tuple[int, int], offset_mapping: AutoTokenizer) -> List[int]:
    """
    This function returns the positions of tokens that correspond to a word,
    given the word range for that word and an Hugging Face offset mapping.

    Parameters:
        word_range (Tuple[int, int]):
            The inclusive-start index and exclusive-end index of a target word
            in the sentence.
        offset_mapping (List[Tuple[int, int]]):
            A list of token index tuples. Each tuple is the start and end index
            of the characters in a sentence that the token represents.

    Returns (List[int]]):
        A list of the positions of the tokens that correspond to the target
        word.
    """
    start_pos, end_pos = word_range

    token_positions = [
        i for i, offset in enumerate(offset_mapping) \
        if offset[0] >= start_pos and offset[1] <= end_pos
    ]
    return token_positions

def word_embedding_from_token_embeddings(
    layer: torch.Tensor, token_positions: List[int]):
    """
    Given a layer of token embeddings and a list of positions of tokens that
    correspond to a word, returns the sum of the embeddings for the tokens that
    correspond to the word.

    Parameters:
        layer (torch.Tensor):
            A tensor of token embeddings for a layer in a Hugging Face model.
        token_positions (List[int]):
            A list of positions of tokens that correspond to a word.

    Returns (torch.Tensor):
        The word embedding for the layer.
    """
    return torch.stack(
        [layer[position] for position in token_positions]
        ).sum(dim=0)


def word_embedding_across_layers(
    hidden_states: torch.Tensor, token_positions: List[int],
    layers_index: List[int]
    ) -> torch.Tensor:
    """
    Averages the word embeddings across specified layers.

    Parameters:
        hidden_states (torch.Tensor):
            A tensor of token embeddings for all layers in a Hugging Face model.
        token_positions (list):
            A list of positions of tokens that correspond to a word.
        layers_index (list):
            A list of index positions for the layers to include in the
            embedding calculation.
    """
    embedding_per_layer = [
        word_embedding_from_token_embeddings(
            hidden_states[layer][0], token_positions
        ) for layer in layers_index]
    return torch.stack(embedding_per_layer).mean(dim=0)


def embeddings_no_context(
    verbs: list[str], embeddings: dict, layers_index: list[int],
    tokenizer: AutoTokenizer, model: AutoModel
    ) -> Dict[str, np.ndarray]:
    """
    Gets the word embeddings for a list of words using a Hugging Face
    transformers model.

    Parameters:
        verbs (list):
            A list of verbs in infinitive form.
        embeddings (dict):
            A dictionary where keys are verbs in infinitive form and values are
            lists of SpaCy embeddings for each instance of the verb in the
            corpus.
        layers_index (list):
            A list of index positions for the layers to include in the
            embedding calculation.
        tokenizer (AutoTokenizer):
            The Hugging Face tokenizer to encode the sentence.
        model (AutoModel):
            The Hugging Face model to create the embeddings.

    Returns:
        returns (dict):
            A dictionary where keys are verbs in infinitive form and values are
            embeddings of a Hugging Face transformer model.
    """
    tokens_positions = [0]

    for verb in verbs:
        inputs = tokenizer(verb, return_tensors='pt')
        outputs = model(**inputs)
        hidden_states = outputs.hidden_states

        word_embedding = word_embedding_across_layers(
            hidden_states, tokens_positions, layers_index
            ).detach().numpy()
        embeddings[verb] = word_embedding

    return embeddings


def embeddings_with_context(
    lemma_and_pos: Dict[str, List[Tuple[int, int]]], sent: str,
    embeddings: Dict[str, List[int]], layers_index: List[int],
    tokenizer: AutoTokenizer, model: AutoModel, is_gpt: bool = False
    ) -> Dict[str, List[np.ndarray]]:
    """
    Firstly, uses a Hugging Face model to derive token embeddings for each
    token in a sentence.
    Secondly, creates word embeddings as the sum of token embeddings that make
    up that word, and does this for each position of each verb found in the
    sentence (as instructed by lemma_and_pos).
    Finally, appends the word embeddings to the embeddings dictionary for this
    model.

    Parameters:
    lemma_and_pos (dict):
        A dictionary where keys are verbs in infinitive form and values are
        lists of word ranges as tuples, where each tuple is the inclusive-start
        index and exclusive-end index of a target word in the sentence.
    sent (str):
        The sentence to process token embeddings for.
    embeddings (dict):
        A dictionary where keys are verbs in infinitive form and values are
        lists of embeddings from a Hugging Face model for each instance of the
        verb in the corpus.
    layers_index (list):
        A list of index positions for the layers to include in the embedding
        calculation.
    tokenizer (AutoTokenizer):
        The Hugging Face tokenizer to encode the sentence.
    model (AutoModel):
        The Hugging Face model to create the embeddings.
    is_gpt (bool):
        GPT tokenizer attaches whitespaces to the first token after the
        whitespace, and this is represented in token offset indexes, so the
        token positions must account for this.

    Returns (dict):
        An updated dictionary of embeddings, where keys are verbs in infinitive
        form and values are lists of embeddings for each instance of the verb
        in the corpus.
    """
    # Embedding for each token of the sentence
    inputs = tokenizer.encode_plus(
        sent, return_tensors='pt', return_offsets_mapping=True,
        max_length=512, truncation=True)
    offset_mapping = inputs.pop('offset_mapping')[0]
    outputs = model(**inputs)
    hidden_states = outputs.hidden_states

    for verb, word_ranges in lemma_and_pos.items():
        for word_range in word_ranges:

            # Simple work-around for GPT tokenizer include whitespace in its
            # offset indexes
            if is_gpt:
                word_range = (max(word_range[0] - 1, 0), word_range[1])

            # Mapping of each target word in the sentence to the tokens that
            # represent it
            tokens_positions = map_words_to_tokens(word_range, offset_mapping)

            # Calculate a word embedding for a word in a sentence as the sum of
            # its tokens, averaged across specified layers
            word_embedding = word_embedding_across_layers(
                hidden_states, tokens_positions, layers_index
                ).detach().numpy()
            embeddings[verb].append(word_embedding)

    return embeddings
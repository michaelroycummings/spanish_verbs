from typing import List, Dict, Tuple
import spacy
import numpy as np
from collections import defaultdict
import subprocess


def load_model(model_name: str, *args, **kwargs) -> spacy.language.Language:
    """ Loads a SpaCy model, and if it doesn't exist, installs it. """
    try:
        return spacy.load(model_name, *args, **kwargs)
    except OSError:
        print(f"Installing Spacy model '{model_name}'")
        command = f"python -m spacy download {model_name}"
        subprocess.run(
            command, shell=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, check=True)
    finally:
        return spacy.load(model_name, *args, **kwargs)


def get_model(
    model_type: str = "medium", task: str = 'all'
    ) -> spacy.language.Language:
    """
    Loads a SpaCy model for Spanish that performs tokenization and word
    embedding via the tok2vec layer, and disables, the other model components
    for faster processing of word embeddings.
    """
    models = {
        "small": "es_core_news_sm",
        "medium": "es_core_news_md",
        "large": "es_core_news_lg",
        "transformer": "es_dep_news_trf"}

    disable_list = {
        "all": [],
        "lemma": ["ner"],
        "pos": ["ner", "attribute_ruler", "lemmatizer"],
        "embedding": [
            "morphologizer", "parser", "senter", "attribute_ruler",
            "lemmatizer", "ner"]
    }
    ## Check if function arguments are valid
    try:
        model_name = models[model_type]
    except KeyError as exc:
        raise ValueError(
            "Invalid model name. Choose from 'small', 'medium', " \
            "'large', or 'transformer'."
            ) from exc
    try:
        disable = disable_list[task]
    except KeyError as exc:
        raise ValueError(
            "Invalid task name. Choose from 'all', 'lemma', 'pos', " \
            "or 'embedding'."
            ) from exc

    return load_model(model_name, disable=disable)


def get_lemma_and_pos(
    sent: spacy.tokens.span.Span) -> Dict[str, List[Tuple[int, int]]]:
    """
    For a single sentence, returns a list of infinitive verbs and the
    inclusive-start index and exclusive-end index positions of each instance of
    that verb's conjugations and infinitive.
    """
    lemma_and_pos = defaultdict(list)
    start_position = sent[0].idx
    for token in sent:
        if token.pos_ in ['VERB', 'AUX']:
            lemma_and_pos[token.lemma_].append((
                token.idx - start_position,
                token.idx + len(token) - start_position))
    return lemma_and_pos


def embeddings_no_context(
    verbs: list, embeddings: Dict[str, np.ndarray], model_type: str,
    model: spacy.language.Language
    ) -> Dict[str, np.ndarray]:
    """
    Gets the word embeddings for a list of words using a SpaCy model.
        Note that this function returns the embedding representation
        of each element in the list. The elements are meant to be words.
        If an element is more than one word, the returned vector will be
        for that sentence.

    Parameters:
        verbs (list):
            A list of verbs in infinitive form.
        embeddings (dict):
            A dictionary where keys are verbs in infinitive form and values
            are lists of SpaCy embeddings for each instance of the verb in the
            corpus.
        model_type (str):
            A string corresponding to the SpaCy model to load.
            One of ['small', 'medium', 'large', 'transformer'].
    Returns:
        returns (dict):
            A dictionary where keys are verbs in infinitive form and values are
            SpaCy embeddings.
    """
    if model_type == 'transformer':
        fetch_embedding = \
            lambda doc: doc._.trf_data.last_hidden_layer_state.data[0]
    else:
        fetch_embedding = lambda doc: doc.vector

    docs = model.pipe(verbs)
    for doc in docs:
        embeddings[doc.text] = fetch_embedding(doc)

    return embeddings


def embeddings_with_context_cnn(
    lemma_and_pos: Dict[str, List[Tuple[int, int]]], sent: str,
    embeddings: Dict[str, List[int]], model: spacy.language.Language
    ) -> Dict[str, List[np.ndarray]]:
    """
    Firstly, uses a SpaCy CNN model to derive token embeddings for each token
    in a sentence.
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
        lists of SpaCy embeddings for each instance of the verb in the corpus.
    model (spacy.language.Language):
        The SpaCy CNN tok2vec model to create the embeddings.

    Returns (dict):
        An updated dictionary of SpaCy embeddings, where keys are verbs in
        infinitive form and values are lists of SpaCy embeddings for each
        instance of the verb in the corpus.
    """
    doc = model(sent)
    for verb, word_ranges in lemma_and_pos.items():
        for word_range in word_ranges:
            tokens_for_word = [
                token for token in doc \
                if token.idx >= word_range[0] \
                and token.idx + len(token) <= word_range[1]]
            word_embedding = np.sum(
                [token.vector for token in tokens_for_word], axis=0)
            embeddings[verb].append(word_embedding)
    return embeddings


def embeddings_with_context_trf(
    lemma_and_pos: Dict[str, List[Tuple[int, int]]], sent: str,
    embeddings: Dict[str, List[int]], model: spacy.language.Language
    ) -> Dict[str, List[np.ndarray]]:
    """
    Firstly, uses a SpaCy CNN model to derive token embeddings for each token
    in a sentence.
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
        lists of SpaCy embeddings for each instance of the verb in the corpus.
    model (spacy.language.Language):
        The SpaCy transformer tok2vec model to create the embeddings.

    Returns (dict):
        An updated dictionary of SpaCy embeddings, where keys are verbs in
        infinitive form and values are lists of SpaCy embeddings for each
        instance of the verb in the corpus.
    """
    doc = model(sent)
    for verb, word_ranges in lemma_and_pos.items():
        for word_range in word_ranges:

            tokens_positions_for_word = [
                i + 1 for i, token in enumerate(doc) \
                if token.idx >= word_range[0] \
                and token.idx + len(token) <= word_range[1]
            ]
            word_embedding = np.sum([
                doc._.trf_data.last_hidden_layer_state.data[i] \
                for i in tokens_positions_for_word], axis=0)
            embeddings[verb].append(word_embedding)
    return embeddings

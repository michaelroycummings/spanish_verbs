from typing import List, Dict, Tuple
import os
import string
import pickle
import numpy as np
from collections import defaultdict
from annoy import AnnoyIndex
from sklearn.metrics import pairwise_distances



def pickle_save(data, loc):
    """ Save a variable to a pickle file. """
    directory = os.path.dirname(loc)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(loc, 'wb') as file:
        pickle.dump(data, file)
    return

def pickle_load(loc):
    """ Load a variable from a pickle file. """
    with open(loc, 'rb') as file:
        return pickle.load(file)


def check_for_punctuation_whitespace(lst):
    dirty_verbs = []
    for verb in lst:
        for char in verb:
            if char in string.punctuation or char in string.whitespace:
                dirty_verbs.append(verb)
                break
    return dirty_verbs


def find_missing_verbs(*args) -> List[str]:
    """
    Finds the verbs that are missing from any of the models.

    Parameters:
    - args:
        A list of embeddings dictionaries, where keys are not used but will
        be model names, and values are dicts where keys are verbs and values are
        not used but will be embedding data arrays of different shapes.
    """
    common_verbs = []
    all_verbs = []

    for embeddings in args:
        for model_data in embeddings.values():
            common_verbs.append(set(model_data.keys()))
            all_verbs.extend(model_data.keys())

    common_verbs = set.intersection(*common_verbs)
    all_verbs = set(all_verbs)
    missing_verbs = list(all_verbs - common_verbs)
    return missing_verbs


def find_dirty_verbs(*args) -> List[str]:
    """
    Finds the verbs that contain punctuation or whitespace.

    Parameters:
    - args:
        A list of embeddings dictionaries, where keys are not used but will
        be model names, and values are dicts where keys are verbs and values are
        not used but will be embedding data arrays of different shapes.
    """
    dirty_verbs = []
    for embeddings in args:
        for model_data in embeddings.values():
            dirty_verbs.extend(
                check_for_punctuation_whitespace(model_data.keys()))
    return list(set(dirty_verbs))


def calc_lengths_per_model(*args) -> Dict[str, List[int]]:
    """
    Returns the lengths of the embeddings for each model.
    Can be used to combine the embedding lengths for with and without context
    embeddings or solely for one.

    Parameters:
    - args:
        A list of embeddings dictionaries, where keys are model names and
        values are dicts where keys are verbs and values are embedding data
        arrays of different shapes.

    Returns:
    - Dict[str, List[int]]:
        A dictionary where keys are model names and values are lists of the
        lengths of the embeddings for each verb in the model.
    """
    lengths = defaultdict(list)
    for embeddings in args:
        for model_name, model_data in embeddings.items():
            try:
                _ = model_data[next(iter(model_data))][0][0] # they are in lists
                embeddings_shape = '2d'
            except IndexError:
                embeddings_shape = '1d'

            if embeddings_shape == '1d':
                lengths[model_name].extend([
                    np.linalg.norm(embedding) \
                    for embedding in model_data.values()])
            else:
                lengths[model_name].extend([
                    np.linalg.norm(embedding) \
                    for embeddings_for_word in model_data.values() \
                    for embedding in embeddings_for_word])

    return lengths


def calc_lengths_per_verb(*args) -> List[str]:
    """
    Find all verbs with zero vectors cross all models, and embedding types
    (context vs contextless).

    Parameters:
    - args:
        A list of embeddings dictionaries, where keys are model names and
        values are dicts where keys are verbs and values are embedding data
        arrays of different shapes.

    Returns:
    - List[str]
        A list of verbs with zero vectors.
    """
    zero_vector_verbs = []

    for embeddings in args:
        for _, model_data in embeddings.items():
            try:
                _ = model_data[next(iter(model_data))][0][0] # they are in lists
                embeddings_shape = '2d'
            except IndexError:
                embeddings_shape = '1d'

            if embeddings_shape == '1d':
                for verb, embedding in model_data.items():
                    if np.linalg.norm(embedding) == 0:
                        zero_vector_verbs.append(verb)
            else:
                for verb, embeddings_for_word in model_data.items():
                    if any(np.linalg.norm(embedding) == 0 \
                    for embedding in embeddings_for_word):
                        zero_vector_verbs.append(verb)

    return list(set(zero_vector_verbs))


def clean_verbs(*args) -> Tuple[Dict[str, Dict[str, np.array]]]:
    """
    Removes dirty verbs and missing verbs from embeddings data objects.

    Parameters:
    - args:
        A list of embeddings dictionaries, where keys are not used but will
        be model names, and values are dicts where keys are verbs and values are
        not used but will be embedding data arrays of different shapes.
    """
    verbs_to_remove = find_dirty_verbs(*args)
    verbs_to_remove.extend(find_missing_verbs(*args))
    verbs_to_remove.extend(calc_lengths_per_verb(*args))
    verbs_to_remove = list(set(verbs_to_remove))

    for embeddings in args:
        for model_data in embeddings.values():
            for verb in verbs_to_remove:
                model_data.pop(verb, None)
    return args


def agg_embeddings_per_model(
    embeddings_context: Dict[str, Dict[str, List[float]]],
    embeddings_contextless: Dict[str, Dict[str, float]],
    ) -> Dict[str, List[float]]:
    """
    Aggregates the embeddings for each infinitive verb (from
    embeddings_contextless) and for all uses of the verb in the corpus
    (from embeddings_context) into one array per verb, for each model.
    """
    embeddings_per_model = defaultdict(list)

    for model_name, verbs_dict in embeddings_context.items():
        embeddings_per_model[model_name].extend([
            embedding for verb_embeddings in verbs_dict.values() \
            for embedding in verb_embeddings
        ])

    for model_name, verbs_dict in embeddings_contextless.items():
        embeddings_per_model[model_name].extend([
            embedding for embedding in verbs_dict.values()
        ])

    return embeddings_per_model


def get_euclidean_distance(emb_1: List[float], emb_2: List[float]) -> float:
    """
    Returns the Euclidean distance between two embeddings.
    """
    return np.linalg.norm(np.array(emb_1) - np.array(emb_2))


def get_cosine_distance(emb_1: List[float], emb_2: List[float]) -> float:
    """
    Returns the cosine distance between two embeddings.
    """
    return 1 - np.dot(emb_1, emb_2) / (
        np.linalg.norm(emb_1) * np.linalg.norm(emb_2))


def get_neighbor_indices(
    embeddings: List[List[float]], num_neighbors: int, num_trees: int = 1
    ) -> List[List[int]]:
    """
    For each embedding in a model, returns the indices of the n nearest
    embeddings.

    Parameters
    ----------
    embeddings : List[List[float]]
        A list of embeddings for a model.
    num_neighbors : int
        The number of nearest neighbors to return.
    num_trees : int
        The number of hyperplane splits to make.
        Larger gives more accurate neigbors estimations at the cost of more
        greater computational load.
        See docs for more info: https://github.com/spotify/annoy

    Return
    ------
    List[List[int]]
        A list of lists of the indices of the n nearest neighbors for
        each embedding in the model.
    """
    dimensionality = len(embeddings[0])

    t = AnnoyIndex(dimensionality, 'euclidean')
    for i, v in enumerate(embeddings):
        t.add_item(i, v)
    t.build(num_trees)

    # t.save('test.ann')
    # t.load('test.ann') # super fast, will just mmap the file

    # Get the neighbor indices for each embedding in a model
    neighbor_indices = []
    for i in range(len(embeddings)):
        indices_for_i = t.get_nns_by_item(
            i,  # index of the embedding
            num_neighbors + 1,
            # search_k, # the number of nodes to search
            )
        neighbor_indices.append(indices_for_i[1:])  # it returns the embedding
        # itself as the first neighbor
    return neighbor_indices


def get_neighbor_distances(
    embeddings: List[List[float]], neighbor_indices: List[List[int]] = None,
    num_neighbors: int = None, **kwargs) -> Tuple[List[List[float]]]:
    """
    For each embedding in a model, returns the Euclidean and Cosine distances
    to the n nearest embeddings.

    Parameters:
        embeddings : List[List[float]]
            A list of embeddings for a model.
        neighbor_indices : List[List[int]]
            A list of lists of the indices of the n nearest neighbors for
            each embedding in the model.
        num_neighbors : int
            The number of nearest neighbors to return.
        kwargs : dict
            Additional keyword arguments to pass to get_neighbor_indices.

    Returns:
        Tuple[List[List[float]]]
            A tuple of lists of the Euclidean and Cosine distances for each
            embedding in the model.
    """
    neighbor_euclideans = []
    neighbor_cosines = []

    # Get indices of the nearest neighbors for each embedding
    if neighbor_indices is None and num_neighbors is None:
            raise ValueError(
                "One of [neighbor_indices, num_neighbors] must be given.")
    elif neighbor_indices is None:
            neighbor_indices = get_neighbor_indices(
                embeddings, num_neighbors, **kwargs)

    # Calculate the distances for each embedding
    for i, neighbors in enumerate(neighbor_indices):
        this_embedding = embeddings[i]

        neighbor_euclideans_for_i = []
        neighbor_cosines_for_i = []

        for neighbor in neighbors:
            neighbor_embedding = embeddings[neighbor]

            neighbor_euclideans_for_i.append(
                get_euclidean_distance(this_embedding, neighbor_embedding))
            neighbor_cosines_for_i.append(
                get_cosine_distance(this_embedding, neighbor_embedding))

        neighbor_euclideans.append(neighbor_euclideans_for_i)
        neighbor_cosines.append(neighbor_cosines_for_i)

    return neighbor_euclideans, neighbor_cosines


def calc_neighbors_per_model(
    embeddings_per_model: Dict[str, List[float]],
    num_neighbors: int = 50, num_trees: int = 100
    ) -> Dict[str, List[List[int]]]:
    """
    For each model, calculates the distances of the n nearest neighbors for each
    embedding.

    Parameters:
    - embeddings_per_model:
        A dictionary where keys are model names and values are lists of
        embeddings for each verb.
    - num_neighbors:
        The number of nearest neighbors to calculate.
    - num_trees:
        The number of hyperplane splits to make.
        Larger gives more accurate neighbors estimations at the cost of more
        greater computational load.
        See docs for more info: https://github.com/spotify/annoy

    Returns:
        Dict[str, Dict[str, List[List[int]]]]
            A dictionary where keys are model names and values are dicts where
            keys are the type of distance metric and values are lists of the
            indices of the n nearest neighbors for each embedding in the model.
    """

    model_neighbors = {}
    for model_name, embeddings in embeddings_per_model.items():

        neighbor_indices = get_neighbor_indices(embeddings, num_neighbors, num_trees)
        euclideans, cosines = get_neighbor_distances(embeddings, neighbor_indices)

        model_neighbors[model_name] = {
            'euclidean': euclideans,
            'cosine': cosines
        }

    return model_neighbors


def calc_distances_context_to_context(
    embeddings_context: Dict[str, Dict[str, List[float]]]
    ) -> Dict[str, List[List[int]]]:
    """
    For each model, calculates the distances of the n nearest neighbors for each
    embedding.

    Parameters:
    - embeddings_context:
        A dictionary where keys are model names and values are dicts where
        keys are verbs and values are lists of embeddings for each use of the
        verb in the corpus.
    # - embeddings_contextless:
    #     A dictionary where keys are model names and values are dicts where
    #     keys are verbs and values are the embedding for each verb's infinitive.
    - num_neighbors:
        The number of nearest neighbors to calculate.
    - num_trees:
        The number of hyperplane splits to make.
        Larger gives more accurate neighbors estimations at the cost of more
        greater computational load.
        See docs for more info: https://github.com/spotify/annoy

    Returns:
        Dict[str, Dict[str, List[List[int]]]]
            A dictionary where keys are model names and values are dicts where
            keys are the type of distance metric and values are lists of the
            indices of the n nearest neighbors for each embedding in the model.
    """

    distances_dict = defaultdict(dict)
    for model_name in embeddings_context.keys():
        euclidean = []
        cosine  = []

        for verb_embeddings in embeddings_context[model_name].values():

            for distance_list, metric in zip(
            [euclidean, cosine], ['euclidean', 'cosine']):

                verb_distances = pairwise_distances(
                    verb_embeddings, metric=metric)
                i_upper = np.triu_indices(verb_distances.shape[0], 1)
                distance_list.extend(verb_distances[i_upper].tolist())

        distances_dict[model_name]['euclidean'] = euclidean
        distances_dict[model_name]['cosine'] = cosine

    return distances_dict


def get_distances_context_to_inf(
    embeddings_context: Dict[str, Dict[str, List[float]]],
    embeddings_contextless: Dict[str, Dict[str, float]],
    ) -> Dict[str, Dict[str, List[float]]]:
    """
    For each model, calculates the distances of each embedding of a verb in
    context to the embedding of the verb's infinitive. Distances are calculated
    using Euclidean and Cosine metrics.

    Parameters
    ----------
    - embeddings_context:
        A dictionary where keys are model names and values are dicts where
        keys are verbs and values are lists of embeddings for each use of the
        verb in the corpus.
    - embeddings_contextless:
        A dictionary where keys are model names and values are dicts where
        keys are verbs and values are the embedding for each verb's infinitive.

    Returns
    -------
    Dict[str, Dict[str, List[float]]]
        A dictionary where keys are model names and values are dicts where
        keys are the type of distance metric and values are lists of the
        distances of each embedding of a verb in context to the embedding of the
        verb's infinitive.
    """
    distances = defaultdict(dict)
    for model_name in embeddings_context.keys():
        euclidean = []
        cosine  = []

        for verb in embeddings_context[model_name].keys():
            infinitive = embeddings_contextless[model_name][verb]

            for verb_in_context in embeddings_context[model_name][verb]:
                euclidean.append(
                    get_euclidean_distance(infinitive, verb_in_context))
                cosine.append(
                    get_cosine_distance(infinitive, verb_in_context))

        distances[model_name]['euclidean'] = euclidean
        distances[model_name]['cosine'] = cosine
    return distances


def get_distances_re_verbs(
    embeddings_context: Dict[str, Dict[str, List[float]]],
    embeddings_contextless: Dict[str, Dict[str, float]],
    ) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    """
    For each model, for euclidean and cosine distance metrics, calculates th
    distribution of distances between each verb and its re counterpart for:
        (1) the contextless, infinitive embeddings.
        (2) each context-containing embedding.

    Parameters
    ----------
    - embeddings_context:
        A dictionary where keys are model names and values are dicts where
        keys are verbs and values are lists of embeddings for each use of the
        verb in the corpus.
    - embeddings_contextless:
        A dictionary where keys are model names and values are dicts where
        keys are verbs and values are the embedding for each verb's infinitive.

    Returns
    -------
    Dict[str, Dict[str, Dict[str, List[float]]]]
        A dictionary where keys are model names and values are dicts where
        keys are the type of distance metric and values are dicts where keys are
        whether the distance is between infinitives or context embeddings, and
        values are lists of the distances of each embedding of a verb in context
        to the embedding of the verb's infinitive.
    """
    distances_dict = defaultdict(dict)
    all_verbs = next(iter(embeddings_contextless.values())).keys()
    verbs_with_re = [verb for verb in all_verbs if f"re{verb}" in all_verbs]

    for model_name in embeddings_context.keys():
        euclidean = defaultdict(list)
        cosine  = defaultdict(list)

        for verb in verbs_with_re:
            re_verb = f"re{verb}"

            # Inf distance
            inf_verb = embeddings_contextless[model_name][verb]
            inf_re_verbs = embeddings_contextless[model_name][re_verb]
            euclidean['inf'].append(
                get_euclidean_distance(inf_verb, inf_re_verbs))
            cosine['inf'].append(
                get_cosine_distance(inf_verb, inf_re_verbs))

            # Context distances
            for context_verb in embeddings_context[model_name][verb]:
                for context_re_verb in embeddings_context[model_name][re_verb]:
                    euclidean['context'].append(
                        get_euclidean_distance(context_verb, context_re_verb))
                    cosine['context'].append(
                        get_cosine_distance(context_verb, context_re_verb))

        distances_dict[model_name]['euclidean'] = euclidean
        distances_dict[model_name]['cosine'] = cosine

    return distances_dict

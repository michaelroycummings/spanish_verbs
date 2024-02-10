from typing import Dict
import numpy as np
import fasttext.util


def get_model(model_filename: str):
    """
    Function loads the spanish fasttext model that should be downloaded
    manually.

    With more time this function would be more robust for automatic downloading
    and moving to a specific directory that would be configurable.
    """
    # fasttext.util.download_model('es', if_exists='ignore')
    return fasttext.load_model(model_filename)

def embeddings_no_context(
    verbs: list[str], embeddings: Dict[str, np.ndarray],
    model: fasttext.FastText._FastText = None
    ) -> Dict[str, np.ndarray]:
    """
    Gets the word embeddings for a list of words using a Fasttext model.

    Parameters:
        verbs (list):
            A list of verbs in infinitive form.
        embeddings (dict):
            A dictionary where keys are verbs in infinitive form and values are
            lists of SpaCy embeddings for each instance of the verb in the
            corpus.

    Returns:
        returns (dict):
            A dictionary where keys are verbs in infinitive form and values are
            SpaCy embeddings.
    """
    for verb in verbs:
        embeddings[verb] = model.get_word_vector(verb)
    return embeddings

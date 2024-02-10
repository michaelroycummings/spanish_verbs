from typing import List, Dict
from collections import defaultdict
import logging
import pickle
import os
import numpy as np
import spacy
from transformers import AutoTokenizer, AutoModel

import fasttext_func
import spacy_func
import hf_func




class EmbeddingsFromCorpus:
    """
    Class to download and process data from a corpus of texts.

    Use the method `derive_data` to download and derive the word counts and
    word embeddings.

    Use the methods below to load the data objects:
        - `load_word_counts`
        - `load_embeddings_with_context`
        - `load_embeddings_no_context`
    """

    def __init__(self, data_dir: str = 'data', log_dir: str = 'logs'):

        # Object to download the Hugging Face corpus
        self.corpus_iterator = None

        # Model Names
        self.model_name_preprocessing_spacy = 'medium'
        self.model_name_embeddings_fasttext = 'models/cc.es.300.bin'
        self.model_name_embeddings_spacy_cnn = 'large'
        self.model_name_embeddings_spacy_trf = 'transformer'
        self.model_name_embeddings_hf_bert = 'bert-base-multilingual-cased'
        self.model_name_embeddings_hf_gpt = 'openai-community/gpt2'

        # Model Objects
        self.model_preprocessing_spacy = None
        self.model_embeddings_fasttext = None
        self.model_embeddings_spacy_cnn = None
        self.model_embeddings_spacy_trf = None
        self.model_embeddings_hf_bert = None
        self.model_embeddings_hf_gpt = None
        self.tokenizer_embeddings_hf_bert = None
        self.tokenizer_embeddings_hf_gpt = None

        # Locations to save Processed Data
        self.loc_data = data_dir
        self.loc_word_counts = os.path.join(self.loc_data, 'word_counts.pkl')
        self.loc_embeddings_with_context = os.path.join(
            self.loc_data, 'embeddings_with_context.pkl')
        self.loc_embeddings_no_context = os.path.join(
            self.loc_data, 'embeddings_no_context.pkl')

        # Data objects to be filled by instance methods
        self.word_counts = None
        self.word_embeddings_with_context = None
        self.word_embeddings_no_context = None

        # Objects for method corpus_derive_counts_and_embeddings
        self.instance_counter = None
        self.sent_counter = None
        self.tf_embedding_layers_index = [-4, -3, -2, -1] # Layers to average
        # when calculating word embeddings


        # Logger
        self.logger = self.create_logger(log_dir)

    def create_logger(self, log_dir: str):
        """
        Create a logger for the class.

        Parameters:
            log_dir (str):
                The directory to save the log file.

        Returns:
            logger (logging.Logger):
                A logger object for the class.
        """
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(funcName)s - '
            '%(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def load_models(self):
        """
        Loads the models for POS Tagging and Lemmatization, Tokenizers, and Word
        Embedding Models.
        Separate function so that class instance can be created without loading
        the models, and so that methods that require the models can be called
        multiple times without loading the models multiple times.
        """
        self.model_preprocessing_spacy = spacy_func.get_model(
            self.model_name_preprocessing_spacy, task='all')
        self.model_embeddings_fasttext = fasttext_func.get_model(
            self.model_name_embeddings_fasttext)
        self.model_embeddings_spacy_cnn = spacy_func.get_model(
            self.model_name_embeddings_spacy_cnn, task='embedding')
        self.model_embeddings_spacy_trf = spacy_func.get_model(
            self.model_name_embeddings_spacy_trf, task='embedding')

        self.tokenizer_embeddings_hf_bert = AutoTokenizer.from_pretrained(
            self.model_name_embeddings_hf_bert)
        self.model_embeddings_hf_bert = AutoModel.from_pretrained(
            self.model_name_embeddings_hf_bert, output_hidden_states=True)

        self.tokenizer_embeddings_hf_gpt = AutoTokenizer.from_pretrained(
            self.model_name_embeddings_hf_gpt)
        self.model_embeddings_hf_gpt = AutoModel.from_pretrained(
            self.model_name_embeddings_hf_gpt, output_hidden_states=True)

    @staticmethod
    def update_word_counts(
        word_counts: Dict[str, int], lemma_and_pos: Dict[str, List]
        ) -> Dict[str, int]:
        """
        Updates a dictionary of word counts with the number of instances
        of each verb or its conjugations in a sentence.

        Parameters:
            word_counts (dict):
                A dictionary where keys are verbs in infinitive form and values
                are the number of instances of that verb in the corpus.
            lemma_and_pos (dict):
                A dictionary where keys are verbs in infinitive form and values
                are lists of word ranges as tuples, where each tuple is the
                inclusive-start index and exclusive-end index of a target word
                in the sentence.

        Returns (dict):
                An updated dictionary of word counts.
        """
        for verb, positions in lemma_and_pos.items():
            word_counts[verb] += len(positions)
        return word_counts


    def update_embeddings_with_context(self,
        embeddings: Dict[str, Dict[str, List[np.ndarray]]],
        text: str, lemma_and_pos: Dict[str, List]
        ) -> Dict[str, Dict[str, List[np.ndarray]]]:
        """
        Updates a dictionary of word embeddings for each word in the corpus.
        See `corpus_derive_counts_embeddings` for details of the word embedding
        object.
        """
        embeddings['spacy_cnn'] = spacy_func.embeddings_with_context_cnn(
            lemma_and_pos, text,
            embeddings['spacy_cnn'],
            self.model_embeddings_spacy_cnn)
        embeddings['spacy_trf'] = spacy_func.embeddings_with_context_trf(
            lemma_and_pos, text,
            embeddings['spacy_trf'],
            self.model_embeddings_spacy_trf)
        embeddings['bert'] = hf_func.embeddings_with_context(
            lemma_and_pos, text,
            embeddings['bert'],
            self.tf_embedding_layers_index,
            self.tokenizer_embeddings_hf_bert,
            self.model_embeddings_hf_bert)
        embeddings['gpt2'] = hf_func.embeddings_with_context(
            lemma_and_pos, text,
            embeddings['gpt2'],
            self.tf_embedding_layers_index,
            self.tokenizer_embeddings_hf_gpt,
            self.model_embeddings_hf_gpt, is_gpt=True)
        return embeddings

    def sent_derive_counts_and_embeddings(self, sent: spacy.tokens.span.Span):
        """
        Processes a single sentence to update word counts and word embeddings
        with context.

        Parameters:
            sent (spacy.tokens.span.Span): The SpaCy sentence to process.

        Returns:
            None
        """
        text = sent.text

        # Get Position Indexes for each Verb in the Sentence
        lemma_and_pos = spacy_func.get_lemma_and_pos(sent)
        if not lemma_and_pos:
            return

        # Update Word Counts
        self.word_counts = self.update_word_counts(
            self.word_counts, lemma_and_pos)

        # Update Word Embeddings with Context
        self.word_embeddings_with_context = self.update_embeddings_with_context(
            self.word_embeddings_with_context, text, lemma_and_pos)
        return

    def corpus_derive_counts_and_embeddings(self, sample_size: int):
        """
        Iterates through the corpus and derives word counts and embeddings with
        context.
        Updates the instance data objects for each sentence so that the data
        can be saved if a crash occurs.

        Parameters:
            sample_size (int):
                The number of sentences to process before stopping.

        Data Objects:
            self.word_counts:
                A dict of word counts. Each word has a count of its instances
                in the corpus.
            self.word_embeddings_with_context:
                A dict of word embedding models. Each model has a dict of words.
                Each word has a 2D vector of its word embedding for that
                model, for each sentence of the corpus that contains the word.
                    Example:
                        self.word_embeddings_with_context ={
                            'spacy_cnn': {
                                'word1': [
                                    [1,2,3,4,5],
                                    [1,2,4,4,5]],
                                'word2': [
                                    [6,7,8,9,10],
                                    [7,7,8,9,9]]},
                            'bert': {
                                ...}}
        Returns:
            None
        """
        # Corpus
        batch_size = 100 # max download size for Hugging Face API is 100
        self.corpus_iterator = hf_func.iter_corpus(batch_size=batch_size)

        # Models: POS and Lemmatizer, Tokenizers, Word Embedding Models
        if self.model_preprocessing_spacy is None:
            self.load_models()

        # Data Objects
        self.word_counts = defaultdict(int)
        self.word_embeddings_with_context = defaultdict(
            lambda: defaultdict(list))

        # Iterate through batches of the corpus instances
        self.instance_counter = 0
        self.sent_counter = 0
        while True:

            # One batch of instances
            batch_text = next(self.corpus_iterator)

            # Check for end of corpus dataset
            if not batch_text: # empty list indicates end of dataset
                break
            self.instance_counter += 1  # will report num instances processed
            sents_processed_this_instance = 0

            # Performs POS Tagging and Lemmatization
            # to find Verbs and their Infinitive Forms
            docs = self.model_preprocessing_spacy.pipe(batch_text)
            sents = [sent for doc in docs for sent in doc.sents]

            self.logger.info(
                "Processing instance #%s with %s sentences.",
                self.instance_counter,
                len(sents))

            for sent_index, sent in enumerate(sents):

                # Derive Word Counts and Embeddings with Context from a sentence
                try:
                    self.sent_derive_counts_and_embeddings(sent)
                    self.sent_counter += 1
                    sents_processed_this_instance += 1
                except Exception:  # pylint: disable=broad-except
                    self.logger.error(
                        "Error when processing sent #%s for instance #%s: %s",
                        sent_index + 1,  # start index at 1; solely for logging
                        self.instance_counter,
                        sent[0:10],
                        exc_info=True)
                    continue

                # Early Stop if Requested Sample Size collected
                if self.sent_counter >= sample_size:
                    break

            self.logger.info(
                "Processed %s sentences for instance #%s.",
                sents_processed_this_instance,
                self.instance_counter)

            # Early Stop if Requested Sample Size collected
            if self.sent_counter >= sample_size:
                break

        return

    def derive_embeddings_no_context(self):
        """
        Derives the word embeddings without context (single words passed to
        word embedding model) for the current set of verbs collected.

        Data Objects:
            embeddings_no_context (dict):
                A dict of word embedding models. Each model has a dict of words.
                Each word has a 1D vector of its word embedding for that model.
                Example:
                    embeddings_no_context = {
                        'word2vec': {
                            'word1': [1,2,3,4,5],
                            'word2': [6,7,8,9,10]},
                        'fasttext': {
                            'word1': [1,2,3,4,5],
                            'word2': [1,2,3,4,5]}}
        """
        if self.model_embeddings_fasttext is None:
            self.load_models()

        verbs = list(self.word_counts.keys())
        embeddings = defaultdict(lambda: defaultdict(list))

        embeddings['fasttext'] = fasttext_func.embeddings_no_context(
            verbs, embeddings['fasttext'], self.model_embeddings_fasttext)
        embeddings['spacy_cnn'] = spacy_func.embeddings_no_context(
            verbs, embeddings['spacy_cnn'],
            self.model_name_embeddings_spacy_cnn,
            self.model_embeddings_spacy_cnn)
        embeddings['spacy_trf'] = spacy_func.embeddings_no_context(
            verbs, embeddings['spacy_trf'],
            self.model_name_embeddings_spacy_trf,
            self.model_embeddings_spacy_trf)
        embeddings['bert'] = hf_func.embeddings_no_context(
            verbs, embeddings['bert'], self.tf_embedding_layers_index,
            self.tokenizer_embeddings_hf_bert,
            self.model_embeddings_hf_bert,)
        embeddings['gpt2'] = hf_func.embeddings_no_context(
            verbs, embeddings['gpt2'], self.tf_embedding_layers_index,
            self.tokenizer_embeddings_hf_gpt,
            self.model_embeddings_hf_gpt,)
        self.word_embeddings_no_context = embeddings
        return

    @staticmethod
    def pickle_save(data, loc):
        directory = os.path.dirname(loc)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(loc, 'wb') as file:
            pickle.dump(data, file)
        return

    @staticmethod
    def pickle_load(loc):
        with open(loc, 'rb') as file:
            return pickle.load(file)

    @staticmethod
    def postprocess_embeddings(
        data_object: Dict[str, Dict[str, List[np.ndarray]]]):
        """
        Converts list of arrays to array.
        """
        for model, model_embeddings in data_object.items():
            for word, word_embeddings in model_embeddings.items():
                data_object[model][word] = np.array(word_embeddings)
        return data_object

    def save_data(self):
        # Cannot pickle a defaultdict
        self.word_embeddings_with_context = dict(
            self.word_embeddings_with_context)
        self.word_embeddings_no_context = dict(
            self.word_embeddings_no_context)

        # Save the three data objects
        for data_obj, loc in (
            [
                self.word_counts,
                self.loc_word_counts
            ],
            [
                self.word_embeddings_with_context,
                self.loc_embeddings_with_context
            ],
            [
                self.word_embeddings_no_context,
                self.loc_embeddings_no_context
            ]
        ):
            try:
                self.pickle_save(data_obj, loc)
            except Exception:  # pylint: disable=broad-except
                self.logger.error("Error when saving to %s", loc, exc_info=True)
        return

    def derive_data(self, sample_size: int = float('inf')):
        """
        Main function to call for deriving word counts and word embeddings from
        the corpus.
        First, iterates through the corpus in a single pass and updates the
        word counts and word embeddings with context after each sentence.
        Then, with the verbs collected, derives the word embeddings without
        context.
        Finally, saves the three data objects.
        """
        self.logger.info(
            "Starting deriving word counts and word embeddings from corpus.")

        # Iterate Through the Corpus and
        # Derive Word Counts and Embeddings with Context
        try:
            self.corpus_derive_counts_and_embeddings(sample_size=sample_size)
        except Exception:  # pylint: disable=broad-except
            self.logger.error(
                "Stopped iterating through the corpus early.", exc_info=True)
        finally:
            self.word_embeddings_with_context = self.postprocess_embeddings(
                self.word_embeddings_with_context)
            self.logger.info(
                "Final counts: processed %s instances and %s sentences.",
                self.instance_counter, self.sent_counter)

        # Derive the word embeddings without context
        # for the current set of verbs collected
        try:
            self.derive_embeddings_no_context()
        except Exception:  # pylint: disable=broad-except
            self.logger.error(
                "Error when deriving embeddings without context.",
                exc_info=True)

        # Save the data
        self.save_data()

        self.logger.info(
            "Finished deriving word counts and word embeddings from corpus.")
        return

    def load_word_counts(self):
        return self.pickle_load(self.loc_word_counts)

    def load_embeddings_with_context(self):
        return self.pickle_load(self.loc_embeddings_with_context)

    def load_embeddings_no_context(self):
        return self.pickle_load(self.loc_embeddings_no_context)

"""Provides an internal interface for working with text features."""

from lightautoml.utils.installation import __validate_extra_deps

__validate_extra_deps('nlp')


__all__ = ['tokenizer', 'dl_transformers', 'sentence_pooling', 'weighted_average_transformer', 'embed_dataset']

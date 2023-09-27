# metrics-predictor
Model-based unit quality prediction for spike sorting

Spike sorting is error-prone and currently requires manual curation to remove false positive and MUA units. This package provides a model-based approach to predict the quality of sorted units based on a combination of quality metrics. An effective way to obtain a good training set for the model is to use an agreement set derived from multiple sorters, as suggested in our [previous paper](https://elifesciences.org/articles/61834). Alternatively, the model can be trained on a manually curated dataset. The model is based on logistic regression along with appropriate data transformations, so the classification results can be easily interpreted in terms of the quality metrics.

This implementation makes use of the functionality of [SpikeInterface](https://github.com/SpikeInterface/spikeinterface).


# Text Embeddings in OpenSearch

This repository is forked from [text-embeddings](https://github.com/jtibshirani/text-embeddings), provides a simple example of how OpenSearch can be used for similarity
search by combining a sentence embedding model with the `knn_vector` field type. As a prerequisite, the [k-NN plugin](https://github.com/opensearch-project/k-NN) must be installed.

**Important note**: Using text embeddings in search is a complex and evolving area. We hope
this example gives a jumping off point for exploration, but it is not a recommendation for
a particular search architecture or implementation.

## Description

The main script `src/main.py` indexes the first ~20,000 questions from the
[StackOverflow](https://github.com/elastic/rally-tracks/tree/master/so)
dataset. Before indexing, each post's title is run through a pre-trained sentence embedding to
produce a [`knn_vector`](https://opensearch.org/docs/latest/search-plugins/knn/knn-index/).

After indexing, the script accepts free-text queries in a loop ("Enter query: ..."). The text is run
through the same sentence embedding to produce a vector, then used to search for similar questions
through [cosine similarity](https://opensearch.org/docs/latest/search-plugins/knn/approximate-knn/).

Currently Google's [Universal Sentence Encoder](https://tfhub.dev/google/universal-sentence-encoder/2) is used
to perform the vector embedding. This is a fully pre-trained model, and no 'fine tuning' is performed
on the StackOverflow dataset.

## Usage

Make sure that `pip` and `python` installed (Python version 3), then install the script's dependencies:

```
pip3 install -r requirements.txt
```

The script assumes that a OpenSearch node is running and able to connect. Instructions on how
to download and run OpenSearch can be found [here](https://opensearch.org/).
Note that **k-NN plugin** is required in order to use the vector functions.

Finally, the script can be run through

```
python3 src/main.py
```

## Example Queries

The following queries return good posts near the top position, despite there not being strong term
overlap between the query and document title:
- "zipping up files" returns "Compressing / Decompressing Folders & Files"
- "determine if something is an IP" returns "How do you tell whether a string is an IP or a hostname"
- "translate bytes to doubles" returns "Convert Bytes to Floating Point Numbers in Python"

Note that in other cases, the results can be noisy and unintuitive. For example, "zipping up files" also assigns high scores to "Partial .csproj Files" and "How to avoid .pyc files?".

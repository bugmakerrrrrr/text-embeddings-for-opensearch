import json
import time

from pathlib import Path

from opensearchpy import OpenSearch
from opensearchpy.helpers import bulk

# Use tensorflow 2 behavior to match the Universal Sentence Encoder
# examples (https://www.kaggle.com/models/google/universal-sentence-encoder/frameworks/tensorFlow2/variations/universal-sentence-encoder/versions/2).
import tensorflow as tf
import tensorflow_hub as hub


##### INDEXING #####
def index_data():
    print("Creating the 'posts' index.")
    client.indices.delete(index=INDEX_NAME, ignore=[404])

    with open(index_file_path) as index_file:
        source = index_file.read().strip()
        client.indices.create(index=INDEX_NAME, body=source)

    docs = []
    count = 0

    with open(data_file_path) as data_file:
        for line in data_file:
            line = line.strip()

            doc = json.loads(line)
            if doc["type"] != "question":
                continue

            docs.append(doc)
            count += 1

            if count % BATCH_SIZE == 0:
                index_batch(docs)
                docs = []
                print("Indexed {} documents.".format(count))

        if docs:
            index_batch(docs)
            print("Indexed {} documents.".format(count))

    client.indices.refresh(index=INDEX_NAME)
    print("Done indexing.")


def index_batch(docs):
    titles = [doc["title"] for doc in docs]
    title_vectors = embed_text(titles)

    requests = []
    for doc, vec in zip(docs, title_vectors):
        request = doc
        request["_op_type"] = "index"
        request["_index"] = INDEX_NAME
        request["title_vector"] = vec
        requests.append(request)
    bulk(client, requests)


##### SEARCHING #####
def run_query_loop():
    while True:
        try:
            handle_query()
        except KeyboardInterrupt:
            print()
            print("finished")
            return


def handle_query():
    query = input("Enter query: ")

    embedding_start = time.time()
    query_vector = embed_text([query])[0]
    embedding_time = time.time() - embedding_start

    knn_query = {
        "knn": {
            "title_vector": {
                "vector": query_vector,
                "k": SEARCH_SIZE
            }
        }
    }

    search_start = time.time()
    response = client.search(
        index=INDEX_NAME,
        body={
            "size": SEARCH_SIZE,
            "query": knn_query,
            "_source": {"includes": ["title", "body"]}
        }
    )
    search_time = time.time() - search_start

    print()
    print("{} total hits.".format(response["hits"]["total"]["value"]))
    print("embedding time: {:.2f} ms".format(embedding_time * 1000))
    print("search time: {:.2f} ms".format(search_time * 1000))
    for hit in response["hits"]["hits"]:
        print("id: {}, score: {}".format(hit["_id"], hit["_score"]))
        print(hit["_source"])
        print()


##### EMBEDDING #####
def embed_text(text):
    vectors = embed(text).numpy()
    return vectors.tolist()


##### MAIN SCRIPT #####
if __name__ == '__main__':
    INDEX_NAME = "posts"
    INDEX_FILE = "data/posts/index.json"
    DATA_FILE = "data/posts/posts.json"
    base_path = Path(__file__).parent
    index_file_path = (base_path / ("../" + INDEX_FILE)).resolve()
    data_file_path = (base_path / ("../" + DATA_FILE)).resolve()

    BATCH_SIZE = 1000
    SEARCH_SIZE = 5
    GPU_LIMIT = 0.5

    print("Loading Universal Sentence Encoder model...")
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    print("Done")

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=int(GPU_LIMIT * 1024))]
        )

    client = OpenSearch(hosts=[{'host': '', 'port': 9200}])

    index_data()
    run_query_loop()

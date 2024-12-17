import os

import numpy as np
import openai

EMBEDDINGS_MODEL = os.environ.get("EMBEDDINGS_MODEL", "text-embedding-3-small")


def main():
    client = openai.Client()

    products = [
        "Search: Ingest your data, and explore Elastic's machine learning and retrieval augmented generation (RAG) capabilities."
        "Observability: Unify your logs, metrics, traces, and profiling at scale in a single platform.",
        "Security: Protect, investigate, and respond to cyber threats with AI-driven security analytics."
        "Elasticsearch: Distributed, RESTful search and analytics.",
        "Kibana: Visualize your data. Navigate the Stack.",
        "Beats: Collect, parse, and ship in a lightweight fashion.",
        "Connectors: Connect popular databases, file systems, collaboration tools, and more.",
        "Logstash: Ingest, transform, enrich, and output.",
    ]

    # Generate embeddings for each product. Keep them in an array instead of a vector DB.
    product_embeddings = []
    for product in products:
        product_embeddings.append(create_embedding(client, product))

    query_embedding = create_embedding(client, "What can help me connect to OpenAI?")

    # Calculate cosine similarity between the query and document embeddings
    similarities = []
    for product_embedding in product_embeddings:
        similarity = np.dot(query_embedding, product_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(product_embedding)
        )
        similarities.append(similarity)

    # Get the index of the most similar document
    most_similar_index = np.argmax(similarities)

    print(products[most_similar_index])


def create_embedding(client, text):
    return client.embeddings.create(input=[text], model=EMBEDDINGS_MODEL, encoding_format="float").data[0].embedding


if __name__ == "__main__":
    main()

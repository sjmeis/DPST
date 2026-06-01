# src/dpst/setup_db.py
import json
import numpy as np
from tqdm.auto import tqdm
import weaviate
import weaviate.classes.config as wc
from weaviate.util import generate_uuid5
from importlib import resources
from collections import Counter

def initialize_database(max_rows=100000):
    """Create the DB collection, parse FineWeb text, embed, and store."""
    from datasets import load_dataset
    
    print("Connecting to local Weaviate instance...")
    client = weaviate.connect_to_local()
    
    if not client.collections.exists("Triples"):
        print("Creating 'Triples' collection schema...")
        client.collections.create(
            name="Triples",
            properties=[
                wc.Property(name="text", data_type=wc.DataType.TEXT),
                wc.Property(name="fiftyk", data_type=wc.DataType.INT),
                wc.Property(name="hundredk", data_type=wc.DataType.INT),
                wc.Property(name="twohundredk", data_type=wc.DataType.INT),
            ],
            vectorizer_config=wc.Configure.Vectorizer.none(),
        )
    
    triples = client.collections.get("Triples")
    
    from dpst.core import embedding_model, get_triples_ie 
    
    print("Streaming sample-10BT from HuggingFaceFW/fineweb...")
    dataset = load_dataset("HuggingFaceFW/fineweb", "sample-10BT", split="train", streaming=True)
    
    all_data = []
    total = 0
    
    for row in dataset:
        if idx == max_rows:
            break
        
        if idx % 100000 == 0 and idx != 0:
            print("{} rows processed.".format(idx))

        if len(row["text"]) > 10000 or row["language"] != "en":
            continue
        
        all_data.append(row["text"])

        if len(all_data) >= 1000:
            print("Extracting triples...")
            res = [get_triples_ie(x) for x in tqdm(all_data)]
            tri = []
            for r in res:
                tri.extend(r)
            
            total += len(tri)
            print("Inserting {} triples...".format(len(tri)))
            embeddings = embedding_model.encode(tri, task="text-matching", truncate_dim=32, max_length=64)

            with triples.batch.dynamic() as batch:
                for i, t in enumerate(tri):
                    obj = {"text":t}
                    vector = embeddings[i]
            
                    batch.add_object(
                        properties=obj,
                        uuid=generate_uuid5(obj),
                        vector=vector
                    )
            print("Finished. Total: {}".format(total))
            del tri[:]
            del tri
            del all_data[:]
            all_data = []

        idx += 1

    print("Extracting triples...")
    res = [get_triples_ie(x) for x in tqdm(all_data)]
    tri = []
    for r in res:
        tri.extend(r)

    total += len(tri)
    print("Inserting {} triples...".format(len(tri)))
    embeddings = embedding_model.encode(tri, task="text-matching", truncate_dim=32, max_length=64)

    with triples.batch.dynamic() as batch:
        for i, t in enumerate(tri):
            obj = {"text":t}
            vector = embeddings[i]

            batch.add_object(
                properties=obj,
                uuid=generate_uuid5(obj),
                vector=vector
            )
    print("Finished. Total: {}".format(total))
    del tri[:]
    del tri
    del all_data[:]
    all_data = []
    
    client.close()
    print("Database seeding completed.")


def run_clustering_and_indexing():
    """Pull vectors, run K-Means, export cluster JSONs, and update Weaviate."""
    from sklearn.cluster import MiniBatchKMeans 

    client = weaviate.connect_to_local()
    triples = client.collections.get("Triples")
    
    print("Fetching vectors from Weaviate...")
    all_ids, all_vectors, all_texts = [], [], []
    for t in tqdm(triples.iterator(include_vector=True)):
        all_ids.append(t.uuid)
        all_vectors.append(t.vector["default"])
    
    x = np.array(all_vectors)
    
    print("Running MiniBatchKMeans (50k, 100k, 200k)...")
    kmeans2 = MiniBatchKMeans(n_clusters=50000, random_state=42, batch_size=8192, max_iter=5, n_init="auto").fit(x)
    kmeans3 = MiniBatchKMeans(n_clusters=100000, random_state=42, batch_size=8192, max_iter=10, n_init="auto").fit(x)
    kmeans4 = MiniBatchKMeans(n_clusters=200000, random_state=42, batch_size=8192, max_iter=10, n_init="auto").fit(x)
    
    data_dir = resources.files("dpst.data")
    
    print(f"Saving cluster assets to: {data_dir}")
    with open(data_dir / "50k.json", 'w') as f:
        json.dump(kmeans2.cluster_centers_.tolist(), f)
    with open(data_dir / "50k_counts.json", 'w') as f:
        json.dump([Counter(kmeans2.labels_)[k] for k in sorted(Counter(kmeans2.labels_))], f)

    with open(data_dir / "100k.json", 'w') as f:
        json.dump(kmeans3.cluster_centers_.tolist(), f)
    with open(data_dir / "100k_counts.json", 'w') as f:
        json.dump([Counter(kmeans3.labels_)[k] for k in sorted(Counter(kmeans3.labels_))], f)

    with open(data_dir / "200k.json", 'w') as f:
        json.dump(kmeans4.cluster_centers_.tolist(), f)
    with open(data_dir / "200k_counts.json", 'w') as f:
        json.dump([Counter(kmeans4.labels_)[k] for k in sorted(Counter(kmeans4.labels_))], f)
        
    print("Updating collection items with cluster allocations...")
    for uid, b, c, d in tqdm(zip(all_ids, kmeans2.labels_, kmeans3.labels_, kmeans4.labels_), total=len(all_ids)):
        props = {"fiftyk": int(b), "hundredk": int(c), "twohundredk": int(d)}
        triples.data.update(uuid=uid, properties=props)
        
    client.close()
    print("Data preparation phase completely finished!")
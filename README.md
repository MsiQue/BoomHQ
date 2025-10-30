# BoomHQ: Learning to Boost Multiple Hybrid Queries on Vector DBMSs

This repository contains the official code for the paper **BoomHQ: Learning to Boost Multiple Hybrid Queries on Vector DBMSs**.  
It includes scripts for **benchmark generation**, **hybrid query construction**, and **learning-based query optimization** on multiple vector database systems.

---

## 🧩 Benchmark Generation

To evaluate hybrid queries that involve **multiple vector columns** and **scalar constraints**, we design a benchmark based on both **vector-only** and **text-to-vector** datasets.  
The following table summarizes all datasets used in our experiments:

| Dataset   | Type | #Samples  | Dim  | Link                                                         |
| --------- | ---- | --------- | ---- | ------------------------------------------------------------ |
| Fungis    | v+s  | 295,938   | 768  | [🔗 Fungis](https://www.kaggle.com/datasets/muhammadfaizan2020/fungi-image-dataset) |
| Sift      | v→s  | 1,000,000 | 128  | [🔗 SIFT1M](http://corpus-texmex.irisa.fr/)                   |
| Glove     | v→s  | 1,183,514 | 100  | [🔗 GloVe](https://nlp.stanford.edu/projects/glove/)          |
| Deep1B    | v→s  | 9,990,000 | 96   | [🔗 Deep1B](https://deepai.org/dataset/deep1b)                |
| Aka_title | s→v  | 361,472   | 768  | [🔗 IMDb Aka-Title](https://datasets.imdbws.com/)             |
| Title     | s→v  | 2,528,312 | 768  | [🔗 IMDb Title](https://datasets.imdbws.com/)                 |
| Aka_name  | s→v  | 901,343   | 768  | [🔗 IMDb Aka-Name](https://datasets.imdbws.com/)              |
| Part      | s→v  | 200,000   | 768  | [🔗 TPC-H Part](https://www.tpc.org/tpch/)                    |
| Partsupp  | s→v  | 800,000   | 768  | [🔗 TPC-H Partsupp](https://www.tpc.org/tpch/)                |
| Orders    | s→v  | 1,500,000 | 768  | [🔗 TPC-H Orders](https://www.tpc.org/tpch/)                  |
| Lineitem  | s→v  | 6,000,000 | 768  | [🔗 TPC-H Lineitem](https://www.tpc.org/tpch/)                |

---

## 🧮 Column Expansion for Hybrid Queries

Since our target task involves **weighted nearest-neighbor queries** across multiple vector columns combined with **scalar filtering conditions**,  
we designed a **column expansion mechanism** to generate realistic hybrid data schemas from the above datasets.

- **Scalar Column Expansion** — implemented in [`gen_scalar_column.py`](./gen_scalar_column.py)  
  This script adds synthetic or derived scalar attributes (e.g., price, rating, category) to simulate hybrid query conditions.

- **Vector Column Expansion** — implemented in [`gen_vector_column.py`](./gen_vector_column.py)  
  This script creates additional semantic vector columns by applying embedding models (e.g., Sentence-BERT, CLIP, or GloVe) to text fields, enabling multi-vector similarity queries.

Together, these scripts construct the foundation of our **BoomHQ benchmark**, enabling reproducible experiments for **Multiple Hybrid Queries (MHQ)**.

---

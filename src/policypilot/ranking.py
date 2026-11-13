"""BM25 + reciprocal rank fusion. Cosine remains the downstream evidence floor."""

import math
import re
from collections import Counter

import numpy as np


def hybrid_order(query, texts, cosine):
    docs = [Counter(re.findall(r"\w+", text.lower())) for text in texts]
    lengths = [sum(d.values()) for d in docs]
    avg = sum(lengths) / max(len(docs), 1) or 1
    lexical = np.zeros(len(docs))
    for term in set(re.findall(r"\w+", query.lower())):
        df = sum(term in doc for doc in docs)
        idf = math.log1p((len(docs) - df + 0.5) / (df + 0.5))
        for i, doc in enumerate(docs):
            tf = doc[term]
            lexical[i] += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * lengths[i] / avg))
    fused = np.zeros(len(docs))
    for scores, sparse in [(cosine, False), (lexical, True)]:
        for rank, i in enumerate(np.argsort(-scores, kind="stable"), 1):
            if not sparse or scores[i] > 0:
                fused[i] += 1 / (60 + rank)
    return np.argsort(-fused, kind="stable")

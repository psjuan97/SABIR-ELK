from elasticsearch import helpers, Elasticsearch
from datetime import datetime
import operator
from functools import reduce

import numpy 
#import shorttext
#from shorttext.classifiers import MaxEntClassifier



# 0 - the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
def generate_features(hits):
    # Build features

    postive_hits = list(filter(lambda x: x["_source"]["class"] == "4", hits))
    negative_hits = list(filter(lambda x: x["_source"]["class"] == "0", hits))

    positive_scores = list(map(lambda x: x["_score"], postive_hits))
    negative_scores = list(map(lambda x: x["_score"], negative_hits))

    if len(negative_scores) == 0:
        features_n = {
            "avg_n": 0,
            "max_n": 0,
            "min_n": 0,
            "sum_n": 0,
            "count_n": 0,
            "phi_n":  0,
        }

        phi_features_n = {
            "phi_avg_n": 0,
            "phi_max_n": 0,
            "phi_min_n":  0,
            "phi_sum_n":  0,
            "phi_count_n":  0,
            "phi_positional_n":  0,
        }
    else:
        features_n = {
            "avg_n": reduce(lambda a, b: a + b, negative_scores) / len(negative_scores),
            "max_n": max(negative_scores),
            "min_n": min(negative_scores),
            "sum_n": sum(negative_scores),
            "count_n": len(negative_scores),
            "phi_n":  phi(negative_hits, hits),
        }

        phi_features_n = {
            "phi_avg_n": features_n["phi_n"] / features_n["avg_n"],
            "phi_max_n":  features_n["phi_n"] / features_n["max_n"],
            "phi_min_n":  features_n["phi_n"] / features_n["min_n"],
            "phi_sum_n":  features_n["phi_n"] / features_n["sum_n"],
            "phi_count_n":  features_n["phi_n"] / features_n["count_n"],
            "phi_positional_n":   phi_positional(negative_hits, hits),
        }


    if len(positive_scores) == 0:
        features_p = {
            "avg_p": 0,
            "max_p": 0,
            "min_p": 0,
            "sum_p": 0,
            "count_p": 0,
            "phi_p": 0
        }

        phi_features_p = {
            "phi_avg_p": 0,
            "phi_max_p":  0,
            "phi_min_p":  0,
            "phi_sum_p": 0,
            "phi_count_p": 0,
            "phi_positinal_p": 0,
        }

    else:
        features_p = {
            "avg_p": reduce(lambda a, b: a + b, positive_scores) / len(positive_scores),
            "max_p": max(positive_scores),
            "min_p": min(positive_scores),
            "sum_p": sum(positive_scores),
            "count_p": len(positive_scores),
            "phi_p": phi(postive_hits, hits)
        }

        phi_features_p = {
            "phi_avg_p":  features_p["phi_p"] / features_p["avg_p"],
            "phi_max_p":  features_p["phi_p"] / features_p["max_p"],
            "phi_min_p":  features_p["phi_p"] / features_p["min_p"],
            "phi_sum_p":  features_p["phi_p"] / features_p["sum_p"],
            "phi_count_p":  features_p["phi_p"] / features_p["count_p"],
            "phi_positinal_p": phi_positional(postive_hits, hits),
        }



    features = {**features_n, **phi_features_n, **features_p, **phi_features_p}


    return features

# We can use memonization tu speed up
def position(hit, rank):
    pos = 1
    # we search the id of the hit in the rank
    for r in rank:
        if hit["_source"]["id"] == r["_source"]["id"]:
            return pos
        pos += 1
    return 0  # never should return 0


def phi(rank_rel, rank_abs):
    return sum(map(lambda x:   (position(x, rank_rel)) / (position(x, rank_abs)), rank_rel))

def phi_positional(rank_rel, rank_abs):
    return sum(map(lambda x:   ( (position(x, rank_rel)) / (position(x, rank_abs))) * x["_score"], rank_rel))


## init elastic

auth = ("elastic", "changeme")
es = Elasticsearch(hosts=["http://localhost:9200"], http_auth=auth)

##READ all training examples to generate features.

import csv
import time


header = "polarity","avg_n","max_n","min_n","sum_n","count_n","phi_n","phi_avg_n","phi_max_n","phi_min_n","phi_sum_n","phi_count_n","phi_positional_n","avg_p","max_p","min_p","sum_p","count_p","phi_p","phi_avg_p","phi_max_p","phi_min_p","phi_sum_p","phi_count_p","phi_positinal_p"
f = open('datasets/test_sdt_features.csv', 'w')
writer = csv.DictWriter(f, fieldnames = header)
writer.writeheader()

start = time.time()

with open('datasets/testdata.manual.2009.06.14.csv') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        start = time.time()

        if row["class"] == '2':
            continue
        #may we can use multisearch and a batch 
        resp = es.search(index="std_train", query={
            "match": {
                "text": row["text"]
            }
        },
            size=26
        )
        hits = resp['hits']['hits']
        #hits.pop(0) only if generating train
        feature = generate_features(hits)
        feature["polarity"] = row["class"]
        end = time.time()
        print(end - start)
        writer.writerow(feature)


end = time.time()
print(end - start)




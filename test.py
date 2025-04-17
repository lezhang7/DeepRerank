import os
def get_qrels_file(name):
    THE_TOPICS = {
        'dl19': 'dl19-passage',
        'dl20': 'dl20-passage',
        'covid': 'beir-v1.0.0-trec-covid-test',
        'arguana': 'beir-v1.0.0-arguana-test',
        'touche': 'beir-v1.0.0-webis-touche2020-test',
        'news': 'beir-v1.0.0-trec-news-test',
        'scifact': 'beir-v1.0.0-scifact-test',
        'fiqa': 'beir-v1.0.0-fiqa-test',
        'scidocs': 'beir-v1.0.0-scidocs-test',
        'nfc': 'beir-v1.0.0-nfcorpus-test',
        'quora': 'beir-v1.0.0-quora-test',
        'dbpedia': 'beir-v1.0.0-dbpedia-entity-test',
        'fever': 'beir-v1.0.0-fever-test',
        'robust04': 'beir-v1.0.0-robust04-test',
        'signal': 'beir-v1.0.0-signal1m-test',
    }
    split = THE_TOPICS.get(name, '')
    split = split.replace('-test', '.test')
    path = 'data/label_file/qrels.' + split + '.txt'  # try to use cache
    if not os.path.exists(path):  # updated to check the correct path variable
        from pyserini.search import get_qrels_file
        return get_qrels_file(THE_TOPICS.get(name, ''))  # download from pyserini using the correct path
    return path  # return the correct path

get_qrels_file('dl19')
breakpoint()
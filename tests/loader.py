import pytest
import json


DATA_SETS_PATHS = ['./tests/data_sets/five_doors.json',
                   './tests/data_sets/corpus.json',
                   './tests/data_sets/git_hub_test.json']


def read(path):
    with open(path, 'r') as f:
        return json.load(f)


def load_data():
    for path in DATA_SETS_PATHS:
        yield read(path)


def generate_data():
    for data in load_data():
        seq_and_result = data.pop('sequences_and_results')
        for elem in seq_and_result:
            yield dict(**data, **elem)


def generate_filtered_data(keys: set):
    for data in filter(lambda x: keys <= x.keys(), generate_data()):
        yield {k: data[k] for k in keys}


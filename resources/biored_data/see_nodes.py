import json
import os
# load in json file

# get file path to full_test_formatted.json
filepath = os.path.join(os.path.dirname(__file__), f'full_test_formatted.json')

with open(filepath) as f:
    data = json.load(f)

# data has a bunch of entries where each entry is a dictionary with the keys 'title' and 'abstract' and 'entities'
# we want to iterate over all entries and extract the entity's type
# we want to just display the set of all entity types
entity_types = set()
for entry in data:
    for entity in entry['entities']:
        entity_types.add(entity['type'])
print(entity_types)

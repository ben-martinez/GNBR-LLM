import graph_utils
import os

# output folder name
output_folder = 'graph-test3-1-output-20241201_235336'
outputs_dir = os.path.join(os.path.dirname(__file__), '../outputs')

outputs_path = os.path.join(outputs_dir, output_folder)

# get list of paths of canonicalized entities files
entities_files = [os.path.join(outputs_path, file) for file in os.listdir(outputs_path) if file.startswith('canonicalized_entities') and file.endswith('.json')]

# similarly get list of paths of canonicalized relationships files
relationships_files = [os.path.join(outputs_path, file) for file in os.listdir(outputs_path) if file.startswith('canonicalized_relationships') and file.endswith('.json')]

# initialize the graph
# get graph path from outputs directory in graphs folder
graph_path = os.path.join(outputs_dir, 'graphs', 'graph-test3-2.json')
graph_utils.create_graph(graph_path)

# add entities to the graph
for entities_file in entities_files:
    # get pubmed id, which is the number that follows the string 'canonicalized_entities_' in the file name
    # the file name is of the form 'canonicalized_entities_{pubmed_id}_{timestamp}.json'
    pubmed_id = entities_file.split('_')[2]

    graph_utils.add_entities_to_graph(graph_path, entities_file, pubmed_id)

# add relationships to the graph
for relationships_file in relationships_files:
    # get pubmed id, which is the number that follows the string 'canonicalized_relationships_' in the file name
    # the file name is of the form 'canonicalized_relationships_{pubmed_id}_{timestamp}.json'
    pubmed_id = relationships_file.split('_')[3]
    graph_utils.add_edges_to_graph(graph_path, relationships_file, pubmed_id)
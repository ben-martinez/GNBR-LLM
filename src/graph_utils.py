import json

# Create a graph JSON file
def create_graph(path):
    # Initialize an empty graph structure
    graph = {
        "nodes": [],
        "edges": []
    }
    
    # Write the structure to the specified file
    with open(path, 'w') as f:
        json.dump(graph, f, indent=4)

    
def add_entities_to_graph(graph_path, entities_path, pubmed_id):
    """
    Adds entities from a JSON file to a graph JSON file.
    ALSO adds the source id to the entity
    
    Args:
        graph_path (str): Path to the graph JSON file.
        entities_path (str): Path to the JSON file containing entity entries.
    """
    # Load the existing graph JSON
    with open(graph_path, 'r') as f:
        graph = json.load(f)
    
    # Load the entities from the input JSON file
    with open(entities_path, 'r') as f:
        entities = json.load(f)

    
    
    # Process each entity and add it to the graph
    for entity in entities:
        entity['source_pubmed_id'] = pubmed_id
        primary_name = entity['primary_name']
        
        # Check if the node for the primary_name already exists
        node = next((node for node in graph['nodes'] if node['id'] == primary_name), None)
        
        if node:
            # Append the entity to the existing node's list of entries
            node['entries'].append(entity)
        else:
            # Create a new node for this primary_name
            new_node = {
                "id": primary_name,
                "type": entity['type'],
                "entries": [entity]  # Start the list with this entity
            }
            graph['nodes'].append(new_node)
    
    # Write the updated graph back to the file
    with open(graph_path, 'w') as f:
        json.dump(graph, f, indent=4)

def add_edges_to_graph(graph_path, edges_path, pubmed_id):
    # Load the existing graph JSON
    with open(graph_path, 'r') as f:
        graph = json.load(f)
    
    # Load the edges from the input JSON file
    with open(edges_path, 'r') as f:
        edges = json.load(f)
    
    # Process each edge and add it to the graph
    for edge in edges:
        edge['source_pubmed_id'] = pubmed_id
        source = edge['source']
        target = edge['target']
        relationship = edge['relationship']
        
        # Check if the edge already exists in the graph
        existing_edge = next(
            (e for e in graph['edges'] if e['source'] == source and e['target'] == target),
            None
        )
        
        if existing_edge:
            # Append the edge information to the existing edge's list of entries
            existing_edge['entries'].append(edge)
        else:
            # Create a new edge with the provided information
            new_edge = {
                "source": source,
                "target": target,
                "relationship": relationship,
                "entries": [edge]  # Start the list with this edge entry
            }
            graph['edges'].append(new_edge)
    
    # Write the updated graph back to the file
    with open(graph_path, 'w') as f:
        json.dump(graph, f, indent=4)
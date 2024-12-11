import json
import re

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

    
def add_aliases_to_node(node, aliases, alias_to_node):
    """
    Add aliases to a node and update the alias_to_node mapping.

    Args:
        node (dict): The node to update.
        aliases (list): List of aliases to add.
        alias_to_node (dict): Mapping from normalized alias to node.
    """
    # Normalize aliases and update the node's alias set
    existing_aliases = set(node.get('aliases', []))
    new_aliases = set(normalize_name(a) for a in aliases)
    all_aliases = existing_aliases.union(new_aliases)
    node['aliases'] = list(all_aliases)
    
    # Update the alias_to_node mapping
    for alias in new_aliases:
        alias_to_node[alias] = node

def add_entities_to_graph(graph_path, entities_path, pubmed_id):
    """
    Adds entities from a JSON file to a graph JSON file.
    Also adds the source PubMed ID to each entity.

    Args:
        graph_path (str): Path to the graph JSON file.
        entities_path (str): Path to the JSON file containing entity entries.
        pubmed_id (str): The source PubMed ID to add to the entities.
    """
    # Load the existing graph JSON
    with open(graph_path, 'r') as f:
        graph = json.load(f)
    
    # Initialize nodes and alias mapping
    nodes = graph.get('nodes', [])
    alias_to_node = {}
    for node in nodes:
        # Ensure aliases is a list
        aliases = node.get('aliases', [])
        if not isinstance(aliases, list):
            aliases = []
        # Include the node id as an alias
        aliases.append(node['id'])
        # Normalize and update aliases
        add_aliases_to_node(node, aliases, alias_to_node)

    # Load the entities from the input JSON file
    with open(entities_path, 'r') as f:
        entities = json.load(f)
    
    # Process each entity and add it to the graph
    for entity in entities:
        entity['source_pubmed_id'] = pubmed_id
        primary_name = entity['primary_name']
        candidates = entity.get('candidates', [])
        
        # Build a list of all possible names to match
        entity_names = [primary_name] + candidates
        normalized_entity_names = [normalize_name(name) for name in entity_names]

        # Try to find a matching node using aliases
        node = None
        for name in normalized_entity_names:
            node = alias_to_node.get(name)
            if node:
                break  # Match found

        if not node:
            # No matching node found, create a new node
            new_node = {
                "id": primary_name,
                "type": entity['type'],
                "aliases": [],  # Will be updated next
                "entries": [entity],
            }
            nodes.append(new_node)
            # Add aliases to the new node and update the mapping
            add_aliases_to_node(new_node, entity_names, alias_to_node)
        else:
            # Matching node found, add entity and update aliases
            node['entries'].append(entity)
            add_aliases_to_node(node, entity_names, alias_to_node)

    # Update the graph's nodes
    graph['nodes'] = nodes

    # Write the updated graph back to the file
    with open(graph_path, 'w') as f:
        json.dump(graph, f, indent=4)

def update_candidate_names(existing_candidates, new_candidates):
    """
    Update the list of candidate names by adding new candidates, avoiding duplicates based on normalized names.

    Args:
        existing_candidates (list): List of existing candidate names.
        new_candidates (list): List of new candidate names to add.

    Returns:
        list: Updated list of candidate names.
    """
    existing_candidates_copy = existing_candidates.copy()
    existing_normalized = set(normalize_name(c) for c in existing_candidates)
    for c in new_candidates:
        norm_c = normalize_name(c)
        if norm_c not in existing_normalized:
            existing_candidates_copy.append(c)
            existing_normalized.add(norm_c)
    return existing_candidates_copy

def add_edges_to_graph(graph_path, edges_path, pubmed_id):
    """
    Adds edges from a JSON file to a graph JSON file.
    Also adds the source PubMed ID to each edge entry.

    Args:
        graph_path (str): Path to the graph JSON file.
        edges_path (str): Path to the JSON file containing edge entries.
        pubmed_id (str): The source PubMed ID to add to the edge entries.
    """
    # Load the existing graph JSON
    with open(graph_path, 'r') as f:
        graph = json.load(f)

    # Initialize edges mapping
    edges = graph.get('edges', [])
    edge_key_to_edge = {}
    for edge in edges:
        # Normalize source, target, and relationship
        normalized_source = normalize_name(edge['source_name'])
        normalized_target = normalize_name(edge['target_name'])
        normalized_relationship = normalize_name(edge['relationship'])
        edge_key = (normalized_source, normalized_target, normalized_relationship)
        edge_key_to_edge[edge_key] = edge

    # Load the edges from the input JSON file
    with open(edges_path, 'r') as f:
        edge_entries = json.load(f)

    # Process each edge entry and add it to the graph
    for edge_entry in edge_entries:
        edge_entry['source_pubmed_id'] = pubmed_id

        # Extract source and target entries
        source_entry = edge_entry['source']
        target_entry = edge_entry['target']
        relationship = edge_entry['relationship']

        # Extract primary names
        source_name = source_entry['primary_name']
        target_name = target_entry['primary_name']

        # Normalize names and relationship
        normalized_source = normalize_name(source_name)
        normalized_target = normalize_name(target_name)
        normalized_relationship = normalize_name(relationship)

        # Build the edge key
        edge_key = (normalized_source, normalized_target, normalized_relationship)

        # Collect candidate names
        source_candidates = source_entry.get('candidates', [])
        target_candidates = target_entry.get('candidates', [])

        if edge_key in edge_key_to_edge:
            # Edge exists
            existing_edge = edge_key_to_edge[edge_key]
            # Append the edge entry to the existing edge's list of entries
            existing_edge['entries'].append(edge_entry)

            # Update source and target candidate names
            existing_edge['source_candidate_names'] = update_candidate_names(
                existing_edge['source_candidate_names'],
                source_candidates
            )
            existing_edge['target_candidate_names'] = update_candidate_names(
                existing_edge['target_candidate_names'],
                target_candidates
            )
        else:
            # Edge does not exist, create new edge
            new_edge = {
                "source_name": source_name,
                "source_candidate_names": source_candidates.copy(),
                "target_name": target_name,
                "target_candidate_names": target_candidates.copy(),
                "relationship": relationship,
                "entries": [edge_entry]  # Start the list with this edge entry
            }
            edges.append(new_edge)
            # Add to edge mapping
            edge_key_to_edge[edge_key] = new_edge
            
    # Update the graph's edges
    graph['edges'] = edges

    # Write the updated graph back to the file
    with open(graph_path, 'w') as f:
        json.dump(graph, f, indent=4)


def normalize_name(name):
    """
    Normalize a name for matching by:
    - Lowercasing
    - Stripping whitespace
    - Removing special characters
    """
    name = name.lower()
    name = name.strip()
    name = re.sub(r'\W+', '', name)  # Remove non-alphanumeric characters
    return name

def get_num_edges(graph_path):
    with open(graph_path, 'r') as f:
        graph = json.load(f)
    return len(graph.get('edges', []))
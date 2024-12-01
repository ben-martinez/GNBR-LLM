from pyvis.network import Network
import json
import os

def visualize_graph_interactive(graph_path, output_path="graph_visualization.html"):
    """
    Visualizes a graph from a JSON file interactively using Pyvis.

    Args:
        graph_path (str): Path to the graph JSON file.
        output_path (str): Path to save the HTML visualization.
    """
    # Load the graph JSON
    with open(graph_path, 'r') as f:
        graph_data = json.load(f)

    # Create a Pyvis Network object
    net = Network(directed=True)

    # Add all nodes first
    node_ids = set()  # Keep track of added node IDs to avoid duplicates
    for node in graph_data['nodes']:
        node_ids.add(node['id'])
        net.add_node(
            node['id'], 
            label=node['id'], 
            title=f"Type: {node['type']}", 
            color="lightblue"
        )

    # Add edges and handle missing nodes
    for edge in graph_data['edges']:
        source = edge['source']
        target = edge['target']
        
        # Add source node if missing
        if source not in node_ids:
            net.add_node(source, label=source, title="Automatically added", color="orange")
            node_ids.add(source)
        
        # Add target node if missing
        if target not in node_ids:
            net.add_node(target, label=target, title="Automatically added", color="orange")
            node_ids.add(target)

        # Add the edge
        net.add_edge(
            source, 
            target, 
            title=edge['relationship'], 
            label=edge['relationship']
        )

    # Generate and open the HTML visualization
    net.show(output_path)


graph_path = os.path.join(os.path.dirname(__file__), '../outputs/graphs/graph-test2.json')
visualize_graph_interactive(graph_path)

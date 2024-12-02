import json
from pyvis.network import Network

def load_knowledge_graph(json_path: str):
    """Load the knowledge graph JSON from a file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def create_interactive_graph(graph_data: dict, output_html: str = "knowledge_graph_interactive.html"):
    """Create an interactive graph visualization from the knowledge graph data using PyVis."""
    # Initialize the PyVis network
    net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white")
    net.force_atlas_2based()  # Use a force-directed layout for better node distribution

    # Optionally set a template (PyVis should handle this by default)
    # net.set_template("template.html")

    # Add nodes to the network
    for node_id, node_data in graph_data.get("nodes", {}).items():
        properties = node_data["properties"]
        entity_label = properties.get("primary_name", node_id)
        description = properties.get("description", "")
        entity_type = properties.get("entity_type", "unknown")
        # Node title is displayed when hovering over it
        title = f"<b>{entity_label}</b><br>Type: {entity_type}<br>{description}"
        net.add_node(
            node_id, 
            label=entity_label, 
            title=title,
            color=get_node_color(entity_type),
            shape="dot",
            size=15
        )

    # Add edges to the network
    for edge_id, edge_data in graph_data.get("edges", {}).items():
        source = edge_data["source_node"]
        target = edge_data["target_node"]
        relationship_type = edge_data["relationship_type"]
        # Edge title is displayed when hovering over it
        title = f"{relationship_type}"
        net.add_edge(
            source, 
            target, 
            label=relationship_type, 
            title=title, 
            color=get_edge_color(relationship_type)
        )

    # Configure physics and layout
    net.repulsion(node_distance=200, central_gravity=0.2, spring_length=200, spring_strength=0.05, damping=0.09)
    
    # Save the network visualization to an HTML file
    net.show(output_html, notebook=False)
    print(f"Graph saved as {output_html}. Open this file in a web browser to view the interactive graph.")

def get_node_color(entity_type: str) -> str:
    """Get a color code for node based on entity_type."""
    colors = {
        "gene": "#FF6F91",
        "protein": "#FF9671",
        "disease": "#FFC75F",
        "chemical_compound": "#F9F871",
        "cell_type": "#D65DB1",
        "biological_process": "#845EC2",
        "organism": "#FFC6FF",
        "unknown": "#FFFFFF"
    }
    return colors.get(entity_type, "#FFFFFF")

def get_edge_color(relationship_type: str) -> str:
    """Get a color code for edge based on relationship_type."""
    colors = {
        "binds_to": "#F1948A",
        "inhibits": "#C0392B",
        "activates": "#27AE60",
        "associated_with": "#2980B9",
        "causes": "#D35400",
        "treats": "#9B59B6",
        "regulates": "#F4D03F",
        "progresses_to": "#E74C3C"
    }
    return colors.get(relationship_type, "#FFFFFF")


if __name__ == "__main__":
    # Load the knowledge graph JSON from a file
    graph_data = load_knowledge_graph("data/knowledge_graph.json")

    # Create an interactive graph visualization and save to an HTML file
    create_interactive_graph(graph_data, "knowledge_graph_interactive.html")
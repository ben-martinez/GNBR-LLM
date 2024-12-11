# README for GNBR 2.0
## Overview
This is a Biomedical Knowledge Graph Tool for extracting, processing, and visualizing relationships between biomedical entities from scientific literature. It combines LLMs (e.g., Llama-3.1-8B), SciSpacy, and PubTator to analyze abstracts, extract entities (e.g., genes, diseases) and relationships (e.g., cause, treat), and populate a dynamic knowledge graph.

## Setting up
Set up a virtual environment and install dependencies in requirements.txt
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Install the SciSpacy Models:
```
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_core_sci_md-0.5.0.tar.gz
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/scispacy-0.5.0.tar.gz
```

Here, we used Cerebras API for Inference. The Cerebras Cloud Python SDK can be access [here](https://github.com/Cerebras/cerebras-cloud-sdk-python). We can set our environment variables:
```
export CEREBRAS_API_KEY='whatever_the_key_is'
```

## Example Use Case
#### Can be tweaked
You can run this script (or something specific to your machine, specific to how the data is set up):
```
python main.py --graph_path ./data/knowledge_graph.json --data_path ./data/data.json --model_name llama3.1-8b
```

## Visualize the Graph
Open index.html in a web browser.

The graph visualization supports:
	•	Node and Edge Filtering: Toggle visibility of nodes and edges by type.
	•	Search Functionality
	•	Zooming

## Logs
Logs are stored in the logs/ directory for monitoring API calls and processing and error handling.
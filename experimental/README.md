# GNBR with LLMs

Check out the graph-test3-2_visualization.html it is a representation of outputs/graphs/graph-test3-2.json which was run with extract_entities.py on 20 abstracts.

Extract_entities.py uses 4 API calls to GPT-4-mini per abstract. 
1. Retrieving: First call asks to get any and all entities, and their description, and if it needs refining
2. Refining: Of the ones that need refining, it focuses on the context of the ambiguous term and gets a better entity name. e.g. 'region 1' becomes 'region 1 of xyz protein'. Hopes to hone in on intra sentence information.
3. Canonicalizing: Asks model to canonicalize to the best of its ability with context
4. Finding Relationships: Gets any and all relationships (currently potentially getting relationships between entities that were not originally retrieved)
5. Reformatting into Canonicalized Relationships: Given the list of canonicalized entities and a basic list of the relationships we ask the model to find the matching canonicalized source or target entity and output a new relationship list where the source and target names are replaced with their full entity information. Note below still may not be working fully.

I sampled 20 abstracts of similar topic from PubMed. I took the top 20 abstracts from the search term: "TNF-alpha signaling in inflammatory bowel disease". This search term was random to get a sample with related topic.

To sample your own abstracts, there's a helper script. You can use the downloader.py file to download abstracts (along with metadata) to a text file directly from PubMed. You just input a text file which has the PubMed ID of the abstract you want on each line.
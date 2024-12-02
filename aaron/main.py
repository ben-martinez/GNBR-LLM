import json
import time
import os
import re
from openai import OpenAI
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from difflib import SequenceMatcher
from collections import defaultdict
import numpy as np
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
from dataclasses import dataclass, asdict
import jsonschema
import time  # For timing the API calls

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EntityInfo:
    name: str
    type: str
    description: Optional[str] = None
    external_ids: Optional[Dict[str, str]] = None

@dataclass
class RelationInfo:
    source_entity: EntityInfo
    target_entity: EntityInfo
    relationship_type: str
    context: Dict
    supporting_text: str
    confidence: float

class LLMProcessor:
    def __init__(self, model: str = "gpt-4o"):
        """Initialize the LLM processor with OpenAI credentials."""
        self.client = OpenAI()
        self.model = model
        self.log_dir = "logs/api_responses"
        os.makedirs(self.log_dir, exist_ok=True)
        self.api_log_path = os.path.join(self.log_dir, "api_calls_log.ndjson")
        
        # Load validation schemas
        self.entity_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "type": {"type": "string"},
                "description": {"type": "string"},
                "external_ids": {
                    "type": "object",
                    "additionalProperties": {"type": "string"}
                }
            },
            "required": ["name", "type"]
        }
        
        self.relation_schema = {
            "type": "object",
            "properties": {
                "source_entity": {"type": "object"},
                "target_entity": {"type": "object"},
                "relationship_type": {"type": "string"},
                "context": {"type": "object"},
                "supporting_text": {"type": "string"},
                "confidence": {"type": "number"}
            },
            "required": ["source_entity", "target_entity", "relationship_type", 
                        "context", "supporting_text", "confidence"]
        }

    def _construct_system_prompt(self) -> str:
        """Construct the system prompt for the LLM."""
        return """You are an expert biomedical knowledge extractor. Your task is to analyze scientific abstracts 
        and extract entities and their relationships. Follow these rules strictly:

        1. Entity Types: protein, gene, disease, chemical_compound, cell_type, biological_process, organism
        2. Relationship Types: inhibits, activates, associated_with, causes, treats, binds_to, regulates
        3. Format all output as valid JSON
        4. Include confidence scores (0-1) for each extraction
        5. Extract experimental context (study type, model system, methods)
        6. Include specific supporting text for each relationship
        7. Be precise with entity names and types
        8. Do not infer relationships not explicitly stated
        9. Include any available entity identifiers (UniProt, MeSH, etc.)

        Output must be in this exact format:
        {
            "entities": [
                {
                    "name": "entity_name",
                    "type": "entity_type",
                    "description": "brief description",
                    "external_ids": {"system": "id"}
                }
            ],
            "relations": [
                {
                    "source_entity": {entity object},
                    "target_entity": {entity object},
                    "relationship_type": "type",
                    "context": {
                        "study_type": "type",
                        "model_system": {"type": "system", "details": "details"},
                        "methods": ["method1", "method2"]
                    },
                    "supporting_text": "exact text from abstract",
                    "confidence": 0.95
                }
            ]
        }
        """

    def _construct_user_prompt(self, abstract_info: Dict) -> str:
        """Construct the user prompt with the abstract and metadata."""
        return f"""Analyze this biomedical abstract and extract entities and relationships:

        Title: {abstract_info['title']}
        Abstract: {abstract_info['abstract']}
        Journal: {abstract_info['journal']}
        Year: {abstract_info['year']}

        Provide all entities and relationships found in the exact JSON format specified."""

    def _validate_entity(self, entity: Dict) -> bool:
        """Validate entity against schema."""
        try:
            jsonschema.validate(instance=entity, schema=self.entity_schema)
            return True
        except jsonschema.exceptions.ValidationError as e:
            logger.warning(f"Entity validation failed: {e}")
            return False

    def _validate_relation(self, relation: Dict) -> bool:
        """Validate relation against schema."""
        try:
            jsonschema.validate(instance=relation, schema=self.relation_schema)
            return True
        except jsonschema.exceptions.ValidationError as e:
            logger.warning(f"Relation validation failed: {e}")
            return False

    def _log_api_response(self, response_data: Dict, abstract_info: Dict, start_time: float, messages: List[Dict]) -> None:
        """Log API response to an NDJSON file."""
        end_time = time.time()
        duration = end_time - start_time
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "duration": duration,
            "abstract_info": abstract_info,
            "model": self.model,
            "messages": messages,
            "api_response": response_data
        }

        # Append the log entry as a line in NDJSON format
        try:
            with open(self.api_log_path, 'a') as f:
                f.write(json.dumps(log_entry) + "\n")
            logger.info(f"API response logged to {self.api_log_path}")
        except Exception as e:
            logger.error(f"Failed to log API response: {e}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _call_llm(self, abstract_info: Dict) -> Dict:
        """Call LLM with retry logic and validation."""
        try:
            start_time = time.time()  # Start timing the API call
            messages = [
                {"role": "system", "content": self._construct_system_prompt()},
                {"role": "user", "content": self._construct_user_prompt(abstract_info)}
            ]
            logger.info("Making API call to OpenAI")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=2000
            )
            
            # Convert response to dictionary for logging
            response_dict = {
                "id": response.id,
                "created": response.created,
                "model": response.model,
                "choices": [{
                    "index": choice.index,
                    "message": {
                        "role": choice.message.role,
                        "content": choice.message.content
                    },
                    "finish_reason": choice.finish_reason
                } for choice in response.choices],
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
            
            # Log the response (including duration and prompts)
            self._log_api_response(response_dict, abstract_info, start_time, messages)
            
            # Parse response content
            try:
                content = response.choices[0].message.content.strip()
                json_block_pattern = re.compile(r'```json\s*(\{.*?\})\s*```', re.DOTALL)

                match = json_block_pattern.search(content)
                if match:
                    json_str = match.group(1)
                    result = json.loads(json_str)
                    return result
                else:
                    result = json.loads(content)
                    return result

            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {e}")
                raise ValueError("Invalid JSON response from LLM")
                
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise

    def _fix_extraction(self, extraction: Dict, abstract_info: Dict) -> Dict:
        """Attempt to fix invalid extraction by asking LLM to correct it."""
        fix_prompt = f"""The previous extraction was invalid. Please fix this extraction to match the required format:

        Previous extraction: {json.dumps(extraction, indent=2)}
        
        Original abstract:
        Title: {abstract_info['title']}
        Abstract: {abstract_info['abstract']}

        Ensure all entities and relations follow the exact schema specified."""

        try:
            start_time = time.time()
            messages = [
                {"role": "system", "content": self._construct_system_prompt()},
                {"role": "user", "content": fix_prompt}
            ]
            logger.info("Making API call to fix extraction format")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
                max_tokens=2000
            )

            # Convert response to dictionary for logging
            response_dict = {
                "id": response.id,
                "created": response.created,
                "model": response.model,
                "choices": [{
                    "index": choice.index,
                    "message": {
                        "role": choice.message.role,
                        "content": choice.message.content
                    },
                    "finish_reason": choice.finish_reason
                } for choice in response.choices],
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
            
            # Log the fix attempt
            self._log_api_response(response_dict, {
                **abstract_info,
                "fix_attempt": True,
                "previous_extraction": extraction
            }, start_time, messages)

            try:
                fixed_result = json.loads(response.choices[0].message.content)
                return fixed_result
            except json.JSONDecodeError:
                raise ValueError("Unable to fix extraction format")
                
        except Exception as e:
            logger.error(f"Fix extraction failed: {e}")
            raise

    def process_abstract(self, abstract_info: Dict) -> Tuple[List[EntityInfo], List[RelationInfo]]:
        """Process a single abstract and return extracted entities and relations."""
        try:
            logger.info(f"Processing abstract {abstract_info.get('pmid', 'N/A')}")
            # Get initial extraction
            extraction = self._call_llm(abstract_info)
            
            # Validate and fix if necessary
            valid = False
            attempts = 0
            while not valid and attempts < 3:
                # Validate entities
                entities_valid = all(self._validate_entity(entity) for entity in extraction.get('entities', []))
                
                # Validate relations
                relations_valid = all(self._validate_relation(relation) for relation in extraction.get('relations', []))
                
                if entities_valid and relations_valid:
                    valid = True
                else:
                    attempts += 1
                    logger.info("Extraction is invalid, attempting to fix extraction format")
                    extraction = self._fix_extraction(extraction, abstract_info)

            if not valid:
                raise ValueError("Unable to obtain valid extraction after multiple attempts")

            # Convert to dataclass objects
            entities = []
            for entity_dict in extraction['entities']:
                entity = EntityInfo(
                    name=entity_dict['name'],
                    type=entity_dict['type'],
                    description=entity_dict.get('description'),
                    external_ids=entity_dict.get('external_ids')
                )
                entities.append(entity)

            relations = []
            for relation_dict in extraction['relations']:
                # Create EntityInfo objects for source and target entities
                source_entity = EntityInfo(
                    name=relation_dict['source_entity']['name'],
                    type=relation_dict['source_entity']['type'],
                    description=relation_dict['source_entity'].get('description'),
                    external_ids=relation_dict['source_entity'].get('external_ids')
                )
                
                target_entity = EntityInfo(
                    name=relation_dict['target_entity']['name'],
                    type=relation_dict['target_entity']['type'],
                    description=relation_dict['target_entity'].get('description'),
                    external_ids=relation_dict['target_entity'].get('external_ids')
                )
                
                relation = RelationInfo(
                    source_entity=source_entity,
                    target_entity=target_entity,
                    relationship_type=relation_dict['relationship_type'],
                    context=relation_dict['context'],
                    supporting_text=relation_dict['supporting_text'],
                    confidence=relation_dict['confidence']
                )
                relations.append(relation)

            return entities, relations

        except Exception as e:
            logger.error(f"Error processing abstract: {e}")
            raise

class KnowledgeGraphUpdater:
    def __init__(self, graph_path: str, entity_aliases_path: str):
        # Initialize empty data structures first
        self.graph = {"nodes": {}, "edges": {}}
        self.entity_aliases = {}
        
        # Store paths for later saving
        self.graph_path = graph_path
        self.entity_aliases_path = entity_aliases_path
        
        # Then try to load from files if they exist
        self.load_graph(graph_path)
        self.load_entity_aliases(entity_aliases_path)
        
        # Build name map and initialize LLM processor
        self.name_to_id_map = self.build_name_map()
        self.llm_processor = LLMProcessor()


        
    def load_graph(self, path: str) -> None:
        """Load existing knowledge graph or create new if missing."""
        try:
            if os.path.exists(path) and os.path.getsize(path) > 0:
                with open(path, 'r') as f:
                    self.graph = json.load(f)
                logger.info(f"Successfully loaded knowledge graph from {path}")
            else:
                logger.info(f"No existing graph found at {path}, initializing new graph")
                # Create the directory if it doesn't exist
                os.makedirs(os.path.dirname(path), exist_ok=True)
                self.save_graph()
        except json.JSONDecodeError as e:
            logger.error(f"Error reading knowledge graph file: {e}")
            logger.info("Initializing new graph")
            self.save_graph()
        except Exception as e:
            logger.error(f"Unexpected error loading knowledge graph: {e}")
            self.save_graph()
            
    def load_entity_aliases(self, path: str) -> None:
        """Load known entity aliases or create new if missing."""
        try:
            if os.path.exists(path) and os.path.getsize(path) > 0:
                with open(path, 'r') as f:
                    self.entity_aliases = json.load(f)
                logger.info(f"Successfully loaded entity aliases from {path}")
            else:
                logger.info(f"No existing aliases found at {path}, initializing empty aliases")
                # Create the directory if it doesn't exist
                os.makedirs(os.path.dirname(path), exist_ok=True)
                self.save_entity_aliases()
        except json.JSONDecodeError as e:
            logger.error(f"Error reading entity aliases file: {e}")
            logger.info("Initializing empty aliases")
            self.save_entity_aliases()
        except Exception as e:
            logger.error(f"Unexpected error loading entity aliases: {e}")
            self.save_entity_aliases()

    def save_graph(self) -> None:
        """Save the current state of the knowledge graph."""
        try:
            with open(self.graph_path, 'w') as f:
                json.dump(self.graph, f, indent=2)
            logger.info(f"Successfully saved knowledge graph to {self.graph_path}")
        except Exception as e:
            logger.error(f"Error saving knowledge graph: {e}")

    def save_entity_aliases(self) -> None:
        """Save the current state of entity aliases."""
        try:
            with open(self.entity_aliases_path, 'w') as f:
                json.dump(self.entity_aliases, f, indent=2)
            logger.info(f"Successfully saved entity aliases to {self.entity_aliases_path}")
        except Exception as e:
            logger.error(f"Error saving entity aliases: {e}")

    def build_name_map(self) -> Dict:
        """Build mapping from all known names (including aliases) to node IDs."""
        name_map = {}
        for node_id, node_data in self.graph["nodes"].items():
            name_map[node_data["properties"]["primary_name"].lower()] = node_id
            for alt_name in node_data["properties"]["alternative_names"]:
                name_map[alt_name.lower()] = node_id
        return name_map

    def find_matching_entity(self, name: str, threshold: float = 0.9) -> Optional[str]:
        """Find matching entity using exact match or fuzzy matching."""
        # Try exact match first
        name_lower = name.lower()
        if name_lower in self.name_to_id_map:
            return self.name_to_id_map[name_lower]
        
        # Try fuzzy matching if no exact match
        for known_name, node_id in self.name_to_id_map.items():
            similarity = SequenceMatcher(None, name_lower, known_name).ratio()
            if similarity >= threshold:
                return node_id
        
        return None

    def create_node(self, entity_info: Dict) -> str:
        """Create a new node with proper ID and metadata."""
        node_id = f"node_{len(self.graph['nodes'])}"
        self.graph["nodes"][node_id] = {
            "type": "string",
            "properties": {
                "entity_type": entity_info["type"],
                "primary_name": entity_info["name"],
                "alternative_names": [],
                "external_ids": {},
                "description": entity_info.get("description", ""),
                "last_updated": datetime.now().isoformat(),
                "creation_date": datetime.now().isoformat()
            }
        }
        # Update name mapping
        self.name_to_id_map[entity_info["name"].lower()] = node_id
        return node_id

    def create_or_update_edge(self, source_id: str, target_id: str, relation_info: Dict) -> str:
        """Create new edge or update existing one with new evidence."""
        # Create unique edge identifier
        edge_key = f"{source_id}_{target_id}_{relation_info['relationship_type']}"
        
        if edge_key not in self.graph["edges"]:
            # Create new edge
            self.graph["edges"][edge_key] = {
                "type": "string",
                "source_node": source_id,
                "target_node": target_id,
                "relationship_type": relation_info["relationship_type"],
                "evidence": [],
                "aggregated_metadata": {
                    "total_papers": 0,
                    "earliest_evidence": None,
                    "latest_evidence": None,
                    "evidence_strength": 0.0,
                    "contradictory_evidence": False
                },
                "last_updated": datetime.now().isoformat()
            }
        
        # Add new evidence
        evidence = {
            "paper_id": relation_info["paper_id"],
            "citation_metadata": relation_info["citation_metadata"],
            "experimental_context": relation_info["experimental_context"],
            "statistical_evidence": relation_info.get("statistical_evidence", {}),
            "extracted_text": relation_info["extracted_text"],
            "extraction_confidence": relation_info["confidence"],
            "last_verified": datetime.now().isoformat()
        }
        
        # Check for duplicate evidence
        if not self._is_duplicate_evidence(edge_key, evidence):
            self.graph["edges"][edge_key]["evidence"].append(evidence)
            self._update_edge_metadata(edge_key)
        
        return edge_key

    def _is_duplicate_evidence(self, edge_key: str, new_evidence: Dict) -> bool:
        """Check if this evidence is already recorded for this edge."""
        existing_evidences = self.graph["edges"][edge_key]["evidence"]
        for evidence in existing_evidences:
            if evidence["paper_id"] == new_evidence["paper_id"]:
                return True
        return False

    def _update_edge_metadata(self, edge_key: str):
        """Update aggregated metadata for an edge."""
        edge = self.graph["edges"][edge_key]
        evidences = edge["evidence"]
        
        edge["aggregated_metadata"].update({
            "total_papers": len(evidences),
            "earliest_evidence": min(e["citation_metadata"]["year"] for e in evidences if e["citation_metadata"]["year"]),
            "latest_evidence": max(e["citation_metadata"]["year"] for e in evidences if e["citation_metadata"]["year"]),
            "evidence_strength": float(np.mean([e["extraction_confidence"] for e in evidences])),
            "last_updated": datetime.now().isoformat()
        })

    def process_abstract(self, abstract_info: Dict) -> List[Dict]:
        """Process a single abstract and update the knowledge graph."""
        # Extract entities and relationships using LLM
        entities, relations = self.llm_processor.process_abstract(abstract_info)
        
        updates = []
        for relation in relations:
            # Find or create source node
            source_id = self.find_matching_entity(relation.source_entity.name)
            if not source_id:
                source_id = self.create_node(asdict(relation.source_entity))
            
            # Find or create target node
            target_id = self.find_matching_entity(relation.target_entity.name)
            if not target_id:
                target_id = self.create_node(asdict(relation.target_entity))
            
            # Create or update edge
            edge_id = self.create_or_update_edge(source_id, target_id, {
                "relationship_type": relation.relationship_type,
                "paper_id": abstract_info["pmid"],
                "citation_metadata": {
                    "title": abstract_info["title"],
                    "authors": abstract_info["authors"],
                    "journal": abstract_info["journal"],
                    "year": abstract_info["year"]
                },
                "experimental_context": relation.context,
                "extracted_text": relation.supporting_text,
                "confidence": relation.confidence
            })
            
            updates.append({
                "edge_id": edge_id,
                "source_id": source_id,
                "target_id": target_id,
                "action": "created" if edge_id not in self.graph["edges"] else "updated"
            })
        
        return updates

# Example usage:
if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
   
    updater = KnowledgeGraphUpdater(
        graph_path="data/knowledge_graph.json",
        entity_aliases_path="data/entity_aliases.json",
    )
    
    # load all the abstracts from the json file data.json
    
    with open("data.json", "r") as f:
        data = json.load(f)
    
    for abstract_info in data:
        try:
            # Process abstract and update graph
            updates = updater.process_abstract(abstract_info)
            logger.info(f"Successfully processed abstract with {len(updates)} updates")
            
            # Save updated graph
            updater.save_graph()
            logger.info("Successfully saved updated knowledge graph")
            
        except Exception as e:
            logger.error(f"Error processing abstract: {e}")
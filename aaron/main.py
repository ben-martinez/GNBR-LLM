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
import argparse
from tenacity import retry, stop_after_attempt, wait_exponential
from dataclasses import dataclass, asdict
import jsonschema
import requests
import spacy
import scispacy
from scispacy.linking import EntityLinker
from scispacy.abbreviation import AbbreviationDetector
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from difflib import SequenceMatcher
from collections import defaultdict
import numpy as np
import logging
from dataclasses import dataclass, asdict
import jsonschema
import time
from tqdm import tqdm
from src.models.cerebras_inference import CerebrasInference
from cerebras.cloud.sdk import Cerebras

# set up logging
# config logging to write to a file
def setup_logging(log_dir='logs'):
    """Set up logging to write to file only, suppressing console output."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'knowledge_graph_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file)  # Log only to file
        ]
    )
    return log_file

log_file = setup_logging()
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

class PubTatorAPI:
    def __init__(self, base_url="https://www.ncbi.nlm.nih.gov/research/pubtator3-api/"):
        self.base_url = base_url

    def find_entity_id(self, entity_name: str, limit: int = 5) -> List[str]:
        """Find entity IDs in PubTator for a given entity name."""
        url = f"{self.base_url}entity/autocomplete/"
        params = {"query": entity_name, "limit": limit}
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return [item["id"] for item in data if "id" in item]

    def find_related_entities(self, entity_id: str, relation_type: str, entity_type: str, limit: int = 5):
        """Find related entities in PubTator for a given entity ID and relation type."""
        url = f"{self.base_url}relations"
        params = {"e1": entity_id, "type": relation_type, "e2": entity_type, "limit": limit}
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json().get("relations", [])

    def search(self, query: str, page: int = 1):
        """Search PubTator for a given query."""
        url = f"{self.base_url}search/"
        params = {"text": query, "page": page}
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()

class LLMProcessor:
    def __init__(self, model: str = "llama3.1-8b"):
        """Initialize the LLM processor with OpenAI credentials."""
        # api key here
        self.client = Cerebras(api_key=os.environ.get("CEREBRAS_API_KEY"))
        self.model = model
        self.log_dir = "logs/api_responses"
        os.makedirs(self.log_dir, exist_ok=True)
        self.api_log_path = os.path.join(self.log_dir, "api_calls_log.ndjson")
        
        # load validation schemas
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
        and extract exclusively biomedical entities and their relationships of the designated types only. Follow these rules strictly:

        1. Entity Types: GENE, DISEASE, CHEMICAL, GENETIC VARIANT (Protein Mutation and DNA Mutation, SNP), SPECIES, PROTEIN
        2. Relationship Types: ASSOCIATE, CAUSE, COMPARE, COTREAT, DRUG_INTERACT, INHIBIT, INTERACT, NEGATIVE_CORRELATE, POSITIVE_CORRELATE, PREVENT, STIMULATE, TREAT, SUBSET
        3. Format all output as valid JSON
        4. Include confidence scores (0-1) for each relation extraction
        5. Extract experimental context (study type, model system, methods)
        6. Include specific supporting text for each relationship
        7. Be precise with entity names and types
        8. Do not infer relationships not stated in the abstract
        9. Include any available entity identifiers (UMLS, etc.) that you have found from external sources and specify

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
        return f"""Analyze this biomedical abstract and extract biomedical entities and their relationships:

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
            
            # log the response (including duration and prompts)
            self._log_api_response(response_dict, abstract_info, start_time, messages)
            
            # parse response content
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

            # convert to dict for logging
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
            
            # log the fix attempt
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
            # get initial extraction
            extraction = self._call_llm(abstract_info)
            
            # validate and fix (if necessary)
            valid = False
            attempts = 0
            while not valid and attempts < 3:
                # validate entities
                entities_valid = all(self._validate_entity(entity) for entity in extraction.get('entities', []))
                
                # validate relations
                relations_valid = all(self._validate_relation(relation) for relation in extraction.get('relations', []))
                
                if entities_valid and relations_valid:
                    valid = True
                else:
                    attempts += 1
                    logger.info("Extraction is invalid, attempting to fix extraction format")
                    extraction = self._fix_extraction(extraction, abstract_info)

            if not valid:
                raise ValueError("Unable to obtain valid extraction after multiple attempts")

            # convert to dataclass objs
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
        # Initialize empty data structs
        self.graph = {"nodes": {}, "edges": {}}
        self.entity_aliases = {}
        
        # Store paths for later saving
        self.graph_path = graph_path
        self.entity_aliases_path = entity_aliases_path
        
        # Try to load from files if they exist
        self.load_graph(graph_path)
        self.load_entity_aliases(entity_aliases_path)
        
        # Build name map and initialize LLM processor
        self.name_to_id_map = self.build_name_map()
        self.llm_processor = LLMProcessor()
        
        # Initialize SciSpacy pipeline
        self.nlp = spacy.load("en_core_sci_md")
        self.nlp.add_pipe("abbreviation_detector")
        self.nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
        self.linker = self.nlp.get_pipe("scispacy_linker")
        
        # Initialize PubTator 3.0 API
        self.pubtator_api = PubTatorAPI()

        
    def load_graph(self, path: str) -> None:
        """Load existing knowledge graph or create new if missing."""
        try:
            if os.path.exists(path) and os.path.getsize(path) > 0:
                with open(path, 'r') as f:
                    self.graph = json.load(f)
                self.graph.setdefault("nodes", {})
                self.graph.setdefault("edges", {})
                logger.info(f"Successfully loaded knowledge graph from {path}")
            else:
                logger.info(f"No existing graph found at {path}, initializing new graph")
                # create the directory if it doesn't exist
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
            for alt_name in node_data["properties"].get("alternative_names", []):
                name_map[alt_name.lower()] = node_id
        return name_map

    def find_matching_entity(self, entity: EntityInfo, threshold: float = 0.5, max_candidates: int = 5) -> Optional[str]:
        name_lower = entity.name.lower()
        entity_type = entity.type

        # Exact match in primary or alternative names
        if name_lower in self.name_to_id_map:
            node_id = self.name_to_id_map[name_lower]
            node_data = self.graph["nodes"][node_id]
            if node_data["properties"]["entity_type"] == entity_type:
                logger.info(f"Exact match found for entity '{entity.name}' with node_id '{node_id}'")
                return node_id
            else:
                logger.warning(f"Type mismatch for entity '{entity.name}' (found type: {node_data['properties']['entity_type']})")

        # Search through nodes for a fuzzy match (including alternative names)
        candidate_entities = []
        for node_id, node_data in self.graph["nodes"].items():
            if node_data["properties"]["entity_type"] != entity_type:
                continue  # Only check same type
            known_names = [node_data["properties"]["primary_name"].lower()] + \
                        [alt_name.lower() for alt_name in node_data["properties"].get("alternative_names", [])]
            for known_name in known_names:
                similarity = SequenceMatcher(None, name_lower, known_name).ratio()
                if similarity >= threshold:
                    candidate_entities.append({
                        "entity_id": node_id,
                        "name": node_data["properties"]["primary_name"],
                        "type": node_data["properties"]["entity_type"]
                    })
                    break  # Avoid duplicate candidates for the same node

        # If candidates found, use LLM disambiguation
        if candidate_entities:
            match_id = self.llm_entity_disambiguation(entity, candidate_entities)
            if match_id:
                logger.info(f"LLM disambiguation matched '{entity.name}' to node_id '{match_id}'")
                return match_id

        logger.info(f"No match found for entity '{entity.name}'")
        return None


    def llm_entity_disambiguation(self, new_entity: EntityInfo, candidate_entities: List[Dict]) -> Optional[str]:
        """
        Use the LLM to determine if the new entity matches any of the candidate entities.
        Return the entity_id of the matching entity, or None if no match.
        """
        print(f"Disambiguating entity: {new_entity.name} using Cerebras.")
        try:
            # Construct the prompt
            prompt = "Your task is to determine if the following new entity matches any of the existing entities.\n\n"
            prompt += "New Entity:\n"
            prompt += json.dumps(asdict(new_entity), indent=2)
            prompt += "\n\nExisting Entities:\n"
            for idx, candidate in enumerate(candidate_entities):
                prompt += f"{idx+1}.\n"
                entity_data = {
                    "entity_id": candidate["entity_id"],
                    "name": candidate["name"],
                    "type": candidate["type"],
                    "description": candidate.get("description", ""),
                    "external_ids": candidate.get("external_ids", {})
                }
                prompt += json.dumps(entity_data, indent=2)
                prompt += "\n"
            prompt += "\nDetermine whether the new entity is the same as any of the existing entities."
            prompt += " Consider the entities to be the same if they refer to the same real-world object or concept."
            prompt += " Pay attention to the entity names, descriptions, and external IDs."
            prompt += " If it matches, output the 'entity_id' of the matching entity."
            prompt += " If it does not match any existing entities, output 'No Match'."
            prompt += "\n\nReturn your answer in JSON format as:\n"
            prompt += '{"match": "entity_id"}\n'
            prompt += "or\n"
            prompt += '{"match": "No Match"}'

            logger.info("Calling Cerebras LLM for entity disambiguation")
            response = self.cerebras_client.chat.completions.create(
                model="llama3.1-8b",  # Replace with your desired Cerebras model
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=100
            )
            print(f"Cerebras LLM response: {response}")
            
            # Get the response content
            content = response.message.content
            # Parse the response
            try:
                result = json.loads(content.strip())
                match = result.get('match')
                if match and match != 'No Match':
                    return match  # return back matching entity_id
                else:
                    return None
            except json.JSONDecodeError:
                logger.error("Failed to parse LLM response as JSON")
                return None
        except Exception as e:
            logger.error(f"Error during LLM entity disambiguation: {e}")
            return None

    def create_node(self, entity_info: Dict) -> str:
        # Final check for existing nodes before creation
        node_id = self.find_matching_entity(EntityInfo(**entity_info))
        if node_id:
            logger.info(f"Found a match during final check, skipping node creation for '{entity_info['name']}'")
            return node_id

        node_id = f"node_{len(self.graph['nodes'])}"
        self.graph["nodes"][node_id] = {
            "type": "string",
            "properties": {
                "entity_type": entity_info["type"],
                "primary_name": entity_info["name"],
                "alternative_names": [],
                "external_ids": entity_info.get("external_ids", {}),
                "description": entity_info.get("description", ""),
                "last_updated": datetime.now().isoformat(),
                "creation_date": datetime.now().isoformat()
            }
        }
        # Add to name mapping
        self.name_to_id_map[entity_info["name"].lower()] = node_id
        logger.info(f"Created new node '{node_id}' for entity '{entity_info['name']}'")
        return node_id


    def update_node(self, node_id: str, entity_info: EntityInfo) -> None:
        """Update an existing node with new information from entity_info."""
        node = self.graph["nodes"][node_id]
        properties = node["properties"]
        # Update description if the new one is longer
        if entity_info.description:
            if not properties.get("description") or len(entity_info.description) > len(properties["description"]):
                properties["description"] = entity_info.description
        # Update external_ids
        if entity_info.external_ids:
            existing_external_ids = properties.get("external_ids", {})
            existing_external_ids.update(entity_info.external_ids)
            properties["external_ids"] = existing_external_ids
        # Add alternative names
        alternative_names = properties.get("alternative_names", [])
        if entity_info.name != properties["primary_name"] and entity_info.name not in alternative_names:
            alternative_names.append(entity_info.name)
        properties["alternative_names"] = alternative_names
        # Update last_updated
        properties["last_updated"] = datetime.now().isoformat()
        # Update the name_to_id_map with the new alternative names
        self.name_to_id_map[entity_info.name.lower()] = node_id

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
        
        # add new evidence
        evidence = {
            "paper_id": relation_info["paper_id"],
            "citation_metadata": relation_info["citation_metadata"],
            "experimental_context": relation_info["experimental_context"],
            "statistical_evidence": relation_info.get("statistical_evidence", {}),
            "extracted_text": relation_info["extracted_text"],
            "extraction_confidence": relation_info["confidence"],
            "last_verified": datetime.now().isoformat()
        }
        
        # check for duplicate evidence
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
        """Update aggregated metadata for a given edge."""
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
        try:
            # Step 1: Extract entities and relationships with LLM
            entities, relations = self.llm_processor.process_abstract(abstract_info)
            logger.info(f"Extracted {len(entities)} entities and {len(relations)} relationships.")

            # Step 2: PubTator normalization
            for ent in entities:
                logger.debug(f"Looking up PubTator ID for entity: {ent.name}")
                try:
                    entity_ids = self.pubtator_api.find_entity_id(ent.name)
                    if entity_ids:
                        ent.external_ids = ent.external_ids or {}
                        ent.external_ids["PubTatorID"] = entity_ids[0]
                        logger.debug(f"Found PubTator ID {entity_ids[0]} for {ent.name}")
                except requests.RequestException as e:
                    logger.warning(f"Failed to fetch PubTator ID for {ent.name}: {e}")

            # Fallbacfk step: SciSpacy Linking
            abstract_text = abstract_info['abstract']
            entities = self.enhance_entities_with_scispacy(entities, abstract_text)
            for ent in entities:
                if ent.external_ids and "UMLS" in ent.external_ids:
                    logger.debug(f"SciSpacy assigned UMLS {ent.external_ids['UMLS']} to {ent.name}")
                else:
                    logger.debug(f"No UMLS CUI found for {ent.name}")

            # Optional: Validate Relationships (commented out for now)
            # for relation in relations:
            #     source_id = relation.source_entity.external_ids.get("PubTatorID")
            #     target_id = relation.target_entity.external_ids.get("PubTatorID")
            #     if source_id and target_id:
            #         # Validate using PubTator relations
            #         pass

            updates = []
            for relation in relations:
                # process source entity
                source_entity = relation.source_entity
                source_id = self.find_matching_entity(source_entity)
                if source_id:
                    self.update_node(source_id, source_entity)
                else:
                    source_id = self.create_node(asdict(source_entity))

                # process entity in question
                target_entity = relation.target_entity
                target_id = self.find_matching_entity(target_entity)
                if target_id:
                    self.update_node(target_id, target_entity)
                else:
                    target_id = self.create_node(asdict(target_entity))

                # create or update edge
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

        except Exception as e:
            logger.error(f"Error processing abstract {abstract_info.get('pmid', 'unknown')}: {e}")
            raise

    def enhance_entities_with_scispacy(self, entities: List[EntityInfo], text: str) -> List[EntityInfo]:
        doc = self.nlp(text)
        # matching LLM-extracted entities by name to Spacy identified spans
        entity_map = {e.name.lower(): e for e in entities}

        for ent in doc.ents:
            name_lower = ent.text.lower()
            if name_lower in entity_map:
                e = entity_map[name_lower]
                # UMLS candidates
                candidates = ent._.kb_ents
                if candidates:
                    # picking top candidate above a certain threshold
                    best_candidate = max(candidates, key=lambda x: x[1])
                    if best_candidate[1] > 0.8:  # the threshold (0.8 arbitrary for now)
                        cui = best_candidate[0]
                        if e.external_ids is None:
                            e.external_ids = {}
                        e.external_ids["UMLS"] = cui
        return list(entity_map.values())

logger = logging.getLogger(__name__)

# Modify the main execution block to use tqdm
if __name__ == "__main__":
    # parse the CL args
    parser = argparse.ArgumentParser()
    # parser.add_argument("--graph_path", default="./data/knowledge_graph.json",
    #                     help="Path to the knowledge graph JSON file.")
    parser.add_argument("--graph_path", default="./data/test_graph.json",
                        help="Path to the knowledge graph JSON file.")
    parser.add_argument("--data_path", default="./data/data.json",
                        help="Path to the input abstracts JSON file.")
    # parser.add_argument("--model_provider", default="cerebras",
    #                     help="Which model provider to use (cerebras, openai, etc.)")
    parser.add_argument("--model_name", default="llama3.1-8b",
                        help="Name of the model to use")
    args = parser.parse_args()
    
    # logging
    print("Program started.")
    # print(f"Using model provider: {args.model_provider}")
    print(f"Model name: {args.model_name}")
    logger.info(f"Logging to file: {log_file}")

    os.makedirs("data", exist_ok=True)
    
    # Initialize components
    inference = CerebrasInference(model=args.model_name)
    updater = KnowledgeGraphUpdater(
        graph_path=args.graph_path,
        entity_aliases_path="./data/entity_aliases.json",
        # Pass inference object if needed in the constructor or assign after
    )
    updater.llm_processor = inference  # if llm_processor replaced by inference
    
    # updater = KnowledgeGraphUpdater(
    #     graph_path="./data/test_graph.json",
    #     entity_aliases_path="./data/entity_aliases.json",
    # )
    
    
    # Load all the abstracts from the json file tes-data.json
    # with open("test-data.json", "r") as f:
    #     data = json.load(f)
        
    # For big run
    with open("data/data.json", "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} abstracts for processing.")

    
    # Use tqdm for progress tracking
    for abstract_info in tqdm(data, desc="Processing Abstracts", unit="abstract"):
        try:
            print(f"\nProcessing abstract with PMID: {abstract_info.get('pmid', 'N/A')}")
            print(f"Title: {abstract_info['title']}")
            print(f"Abstract: {abstract_info['abstract'][:100]}...")  # Print a snippet of the abstract
            
            # process abstract + update graph
            updates = updater.process_abstract(abstract_info)
            logger.info(f"Successfully processed abstract {abstract_info.get('pmid', 'N/A')} with {len(updates)} updates.")
            
            # save updated graph
            updater.save_graph()
            logger.info("Successfully saved updated knowledge graph")            
        except Exception as e:
            logger.error(f"Error processing abstract {abstract_info.get('pmid', 'N/A')}: {e}")

    logger.info("Finished processing all abstracts.")
    print("Program completed.")
# src/models/cerebras_inference.py

import os
import re
import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import jsonschema
import time
from difflib import SequenceMatcher

from cerebras.cloud.sdk import Cerebras

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

class CerebrasInference:
    def __init__(self, model: str, api_key=None):
        self.model = model
        self.api_key = api_key or os.environ.get("CEREBRAS_API_KEY")
        if not self.api_key:
            logger.error("CEREBRAS_API_KEY not set.")
            raise ValueError("CEREBRAS_API_KEY is required for CerebrasInference.")
        self.client = Cerebras(api_key=self.api_key)
        logger.info(f"CerebrasInference initialized with model: {self.model}")

        # Define JSON schemas for validation
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

        # Log file path for API calls
        self.api_log_path = "./logs/api_calls_log.ndjson"
        os.makedirs(os.path.dirname(self.api_log_path), exist_ok=True)

    def chat_completion(self, messages: List[Dict]) -> str:
        """
        Sends a list of messages to the Cerebras LLM and returns the response content.

        Args:
            messages (list): List of message dictionaries with 'role' and 'content'.

        Returns:
            str: The content of the LLM's response.
        """
        try:
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                temperature=0.3,
                max_tokens=2000
            )
            content = response.choices[0].message.content.strip()
            logger.debug(f"Cerebras LLM response: {content}")
            return content
        except Exception as e:
            logger.error(f"Cerebras API call failed: {e}")
            raise

    def _log_api_response(self, response_content: str, abstract_info: Dict, start_time: float, messages: List[Dict], fix_attempt: bool = False, previous_extraction: Dict = None) -> None:
        """Log API response to an NDJSON file."""
        end_time = time.time()
        duration = end_time - start_time
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "duration": duration,
            "abstract_info": abstract_info,
            "model": self.model,
            "messages": messages,
            "api_response": response_content
        }
        if fix_attempt:
            log_entry["fix_attempt"] = True
            log_entry["previous_extraction"] = previous_extraction

        # Append the log entry as a line in NDJSON format
        try:
            with open(self.api_log_path, 'a') as f:
                f.write(json.dumps(log_entry) + "\n")
            logger.info(f"API response logged to {self.api_log_path}")
        except Exception as e:
            logger.error(f"Failed to log API response: {e}")

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


    def process_abstract(self, abstract_info: Dict) -> Tuple[List[EntityInfo], List[RelationInfo]]:
        """
        Process a single abstract and return extracted entities and relations.

        Args:
            abstract_info (Dict): Dictionary containing abstract details.

        Returns:
            Tuple[List[EntityInfo], List[RelationInfo]]: Extracted entities and relations.
        """
        try:
            logger.info(f"Processing abstract PMID: {abstract_info.get('pmid', 'N/A')}")
            # 1. Construct the prompts
            system_prompt = """You are an expert biomedical knowledge extractor. Your task is to analyze scientific abstracts 
            and extract exclusively biomedical entities and their relationships of the designated types only. Follow these rules strictly:

            1. Entity Types: GENE, PROTEIN, DISEASE, CHEMICAL, GENETIC VARIANT (Protein Mutation and DNA Mutation, SNP), SPECIES
            2. Relationship Types: ASSOCIATE, CAUSE, COMPARE, COTREAT, DRUG_INTERACT, INHIBIT, INTERACT, NEGATIVE_CORRELATE, POSITIVE_CORRELATE, PREVENT, STIMULATE, TREAT, SUBTYPE
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

            user_prompt = f"""Analyze this biomedical abstract and extract biomedical entities and their relationships:

            Title: {abstract_info['title']}
            Abstract: {abstract_info['abstract']}
            Journal: {abstract_info['journal']}
            Year: {abstract_info['year']}

            Provide all entities and relationships found in the exact JSON format specified."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            # 2. Call CerebrasInference to get the response
            start_time = time.time()
            response_content = self.chat_completion(messages)
            self._log_api_response(response_content, abstract_info, start_time, messages)

            # 3. Parse the response
            try:
                # Attempt to extract JSON block if wrapped in code fences
                json_block_pattern = re.compile(r'```json\s*(\{.*?\})\s*```', re.DOTALL)
                match = json_block_pattern.search(response_content)
                if match:
                    json_str = match.group(1)
                else:
                    json_str = response_content

                result = json.loads(json_str.strip())
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {e}")
                raise ValueError("Invalid JSON response from LLM")

            # 4. Validate the extracted data
            entities = result.get("entities", [])
            relations = result.get("relations", [])

            for entity in entities:
                if not self._validate_entity(entity):
                    logger.warning(f"Entity validation failed for: {entity}")
                    raise ValueError("Entity validation failed")

            for relation in relations:
                if not self._validate_relation(relation):
                    logger.warning(f"Relation validation failed for: {relation}")
                    raise ValueError("Relation validation failed")

            # 5. Convert to dataclass instances
            entities_info = [
                EntityInfo(
                    name=entity['name'],
                    type=entity['type'],
                    description=entity.get('description'),
                    external_ids=entity.get('external_ids')
                ) for entity in entities
            ]

            relations_info = [
                RelationInfo(
                    source_entity=EntityInfo(
                        name=relation['source_entity']['name'],
                        type=relation['source_entity']['type'],
                        description=relation['source_entity'].get('description'),
                        external_ids=relation['source_entity'].get('external_ids')
                    ),
                    target_entity=EntityInfo(
                        name=relation['target_entity']['name'],
                        type=relation['target_entity']['type'],
                        description=relation['target_entity'].get('description'),
                        external_ids=relation['target_entity'].get('external_ids')
                    ),
                    relationship_type=relation['relationship_type'],
                    context=relation['context'],
                    supporting_text=relation['supporting_text'],
                    confidence=relation['confidence']
                ) for relation in relations
            ]

            return entities_info, relations_info

        except ValueError as ve:
            logger.error(f"Value error during processing abstract PMID {abstract_info.get('pmid', 'N/A')}: {ve}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during processing abstract PMID {abstract_info.get('pmid', 'N/A')}: {e}")
            raise
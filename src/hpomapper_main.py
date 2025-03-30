import os
import json
import numpy as np
import pandas as pd
import faiss
import boto3
import re
import time
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import logging
from tqdm import tqdm
import fuzzywuzzy.fuzz as fuzz

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("hpomapper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HPOVectorDB:
    """Manages the HPO vector database for similarity search"""
    
    def __init__(self, embedding_file: str = "G2GHPO_metadata.npy"):
        """
        Initialize the HPO Vector Database
        
        Args:
            embedding_file: Path to the numpy file containing HPO embeddings
        """
        self.embedding_file = embedding_file
        self.index = None
        self.hpo_data = None
        self.load_embeddings()
        
    def load_embeddings(self):
        """Load HPO embeddings and create a FAISS index"""
        try:
            # Load the precomputed HPO embeddings
            data = np.load(self.embedding_file, allow_pickle=True).item()
            self.hpo_data = data
            
            # Extract embeddings and create FAISS index
            # First, process the embeddings to ensure they're all valid numpy arrays with uniform shape
            processed_embeddings = []
            valid_items = []
            
            dimension = None  # We'll determine the dimension from the first valid embedding
            
            for i, item in enumerate(data['items']):
                try:
                    # Convert embedding to numpy array if it's not already
                    embed = np.array(item['embedding'], dtype=np.float32)
                    
                    # Check if this is our first valid embedding to determine the dimension
                    if dimension is None:
                        dimension = embed.shape[0]
                    
                    # Verify the embedding has the correct dimension
                    if embed.shape == (dimension,):
                        processed_embeddings.append(embed)
                        valid_items.append(item)
                    else:
                        logger.warning(f"Skipping item {i} with embedding of unexpected shape: {embed.shape}")
                except (ValueError, TypeError) as e:
                    logger.warning(f"Skipping item {i} with invalid embedding: {str(e)}")
            
            # Update the data with only valid items
            self.hpo_data = {
                'items': valid_items,
                'model_id': data.get('model_id', 'unknown'),
                'created_at': data.get('created_at', 'unknown')
            }
            
            # Create a numpy array from the processed embeddings
            embeddings = np.array(processed_embeddings, dtype=np.float32)
            
            # Create and train the index
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings)
            
            logger.info(f"Successfully loaded HPO embeddings with {len(valid_items)} valid terms")
            
            # Report stats on how many items were skipped
            skipped_count = len(data['items']) - len(valid_items)
            if skipped_count > 0:
                logger.warning(f"Skipped {skipped_count} items with invalid embeddings")
                
        except Exception as e:
            logger.error(f"Failed to load HPO embeddings: {str(e)}")
            raise
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """
        Search the HPO database for similar terms
        
        Args:
            query_embedding: The embedding vector of the query
            k: Number of results to return
            
        Returns:
            List of dictionaries containing HPO terms and metadata
        """
        if self.index is None:
            raise ValueError("HPO database not loaded. Call load_embeddings() first.")
        
        # Reshape the query embedding if necessary
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1).astype('float32')
            
        # Perform the search
        distances, indices = self.index.search(query_embedding, k)
        
        # Get the results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.hpo_data['items']):
                item = self.hpo_data['items'][idx]
                results.append({
                    'hpo_id': item['hpo_id'],
                    'name': item['name'],
                    'distance': float(distances[0][i]),
                    'definition': item.get('definition', ''),
                    'synonyms': item.get('synonyms', [])
                })
                
        return results
    
    def update_embeddings(self, hpo_json_path: str, llm_client, output_path: str = "G2GHPO_metadata.npy"):
        """
        Update the HPO embeddings using the latest HPO JSON file
        
        Args:
            hpo_json_path: Path to the HP.json file
            llm_client: LLM client to generate embeddings
            output_path: Path to save the updated embeddings
        """
        # This is a placeholder for the full implementation
        # In a real implementation, this would:
        # 1. Parse the HPO JSON file
        # 2. Extract terms, definitions, and synonyms
        # 3. Generate embeddings for each term using the LLM
        # 4. Save the embeddings to a numpy file
        logger.info("HPO embeddings update not fully implemented")
        pass


import aws_helper

class BedrockLLM:
    """Interface for AWS Bedrock LLM services"""
    
    def __init__(self, model_id: str = "anthropic.claude-v2", region_name: str = None, profile_name: str = None):
        """
        Initialize the Bedrock LLM client
        
        Args:
            model_id: The Bedrock model ID to use
            region_name: AWS region name (optional)
            profile_name: AWS profile name for SSO (optional)
        """
        self.model_id = model_id
        
        # Use the helper to get a properly authenticated Bedrock client
        self.bedrock_client = aws_helper.get_bedrock_client(
            profile_name=profile_name,
            region_name=region_name
        )
        
        logger.info(f"Initialized Bedrock client with model {model_id}")
        
    def extract_clinical_phrases(self, clinical_note: str) -> List[str]:
        """
        Extract clinically relevant phrases from a clinical note
        
        Args:
            clinical_note: The clinical note text
            
        Returns:
            List of extracted clinical phrases
        """
        prompt = f"""
        Extract all clinically relevant phenotypic descriptions from the following clinical note.
        Respond with a list of individual phenotypic findings, one per line.
        Only include observations about the patient's phenotype, not diagnoses, treatments, or family history.
        
        Clinical note:
        {clinical_note}
        
        Phenotypic findings:
        """
        
        response = self._call_bedrock(prompt)
        
        # Extract phrases from the response (one per line)
        phrases = [line.strip() for line in response.split('\n') if line.strip()]
        return phrases
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate an embedding for a given text
        
        Args:
            text: The text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        # This would use a Bedrock embedding model
        # For now, we'll return a simple placeholder
        # In a real implementation, you would use a model like Amazon Titan Embeddings
        raise NotImplementedError("Embedding generation not implemented")
    
    def _call_bedrock(self, prompt: str) -> str:
        """
        Call the Bedrock LLM API
        Args:
        prompt: The prompt to send to the LLM
        Returns:
        The LLM response text
        """
        try:
            if "anthropic.claude-3" in self.model_id.lower():
                # Format for Claude 3 models with messages API
                request_body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 2000,
                    "temperature": 0.5,
                    "top_p": 0.999,
                    "top_k": 250,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt
                                }
                            ]
                        }
                    ]
                }
                
                response = self.bedrock_client.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps(request_body)
                )
                
                response_body = json.loads(response.get('body').read())
                
                # Extract response from Claude 3 message format
                if "content" in response_body and len(response_body["content"]) > 0:
                    # Get the text from the first content item
                    for content_item in response_body["content"]:
                        if content_item.get("type") == "text":
                            return content_item.get("text", "")
                return ""
                
            elif "anthropic.claude" in self.model_id.lower():
                # Format for Claude v2 models
                request_body = {
                    "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
                    "max_tokens_to_sample": 2000,
                    "temperature": 0.5,
                    "top_p": 1,
                    "top_k": 250,
                    "stop_sequences": ["\n\nHuman:"],
                    "anthropic_version": "bedrock-2023-05-31"
                }
                
                response = self.bedrock_client.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps(request_body)
                )
                
                response_body = json.loads(response.get('body').read())
                return response_body.get('completion', '')
                
            elif "meta.llama" in self.model_id.lower():
                # Format for Meta Llama models
                request_body = {
                    "prompt": prompt,
                    "max_gen_len": 512,
                    "temperature": 0.5,
                    "top_p": 0.9
                }
                
                response = self.bedrock_client.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps(request_body)
                )
                
                response_body = json.loads(response.get('body').read())
                return response_body.get('generation', '')
                
            else:
                # Generic format for other models
                request_body = {
                    "inputText": prompt,
                    "textGenerationConfig": {
                        "maxTokenCount": 2000,
                        "temperature": 0.5,
                        "topP": 0.9,
                    }
                }
                
                response = self.bedrock_client.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps(request_body)
                )
                
                response_body = json.loads(response.get('body').read())
                return response_body.get('results', [{}])[0].get('outputText', '')
                
        except Exception as e:
            logger.error(f"Bedrock API call failed: {str(e)}")
            raise

class hpomapper:
    """Main class for HPO term extraction from clinical notes"""
    
    def __init__(self, 
                 llm_client: BedrockLLM, 
                 hpo_db: HPOVectorDB,
                 fuzzy_match_threshold: int = 80):
        """
        Initialize hpomapper
        
        Args:
            llm_client: LLM client for clinical phrase extraction
            hpo_db: HPO vector database for term matching
            fuzzy_match_threshold: Threshold for fuzzy matching (0-100)
        """
        self.llm_client = llm_client
        self.hpo_db = hpo_db
        self.fuzzy_match_threshold = fuzzy_match_threshold
        
    def process_clinical_notes(self, 
                              notes_file: str, 
                              output_file: str = "hpomapper_results.csv") -> pd.DataFrame:
        """
        Process clinical notes and extract HPO terms
        
        Args:
            notes_file: Path to the clinical notes file (CSV with 'patient_id' and 'note' columns)
            output_file: Path to save the results
            
        Returns:
            DataFrame with extracted HPO terms
        """
        # Load clinical notes
        notes_df = pd.read_csv(notes_file)
        
        if 'patient_id' not in notes_df.columns or 'note' not in notes_df.columns:
            raise ValueError("Clinical notes file must have 'patient_id' and 'note' columns")
        
        # Initialize results
        results = []
        
        # Process each clinical note
        for _, row in tqdm(notes_df.iterrows(), total=len(notes_df), desc="Processing clinical notes"):
            patient_id = row['patient_id']
            note = row['note']
            
            # Extract clinical phrases
            phrases = self.llm_client.extract_clinical_phrases(note)
            
            # Match phrases to HPO terms
            for phrase in phrases:
                matches = self.match_phrase_to_hpo(phrase)
                
                if matches:
                    # Take the best match
                    best_match = matches[0]
                    
                    results.append({
                        'patient_id': patient_id,
                        'clinical_phrase': phrase,
                        'hpo_id': best_match['hpo_id'],
                        'hpo_term': best_match['name'],
                        'match_score': 1.0 - best_match['distance'],
                        'definition': best_match.get('definition', '')
                    })
        
        # Create DataFrame from results
        results_df = pd.DataFrame(results)
        
        # Save results to CSV
        if output_file:
            results_df.to_csv(output_file, index=False)
            logger.info(f"Results saved to {output_file}")
        
        return results_df
    
    def match_phrase_to_hpo(self, phrase: str) -> List[Dict]:
        """
        Match a clinical phrase to HPO terms using vector search and fuzzy matching
        
        Args:
            phrase: The clinical phrase to match
            
        Returns:
            List of matching HPO terms with metadata
        """
        # Step 1: Try direct fuzzy matching with HPO terms
        fuzzy_matches = self._fuzzy_match_hpo(phrase)
        
        # If we found good fuzzy matches, return them
        if fuzzy_matches and fuzzy_matches[0]['score'] >= self.fuzzy_match_threshold:
            # Convert to the same format as vector search results
            return [{
                'hpo_id': match['hpo_id'],
                'name': match['name'],
                'distance': 1.0 - (match['score'] / 100.0),
                'definition': match.get('definition', '')
            } for match in fuzzy_matches[:3]]  # Return top 3 fuzzy matches
        
        # Step 2: If no good fuzzy matches, try vector similarity search
        # This would require embedding the phrase and searching the vector database
        # Not implemented here as it requires a specific embedding model
        
        # For now, return the fuzzy matches even if they're below threshold
        if fuzzy_matches:
            return [{
                'hpo_id': match['hpo_id'],
                'name': match['name'],
                'distance': 1.0 - (match['score'] / 100.0),
                'definition': match.get('definition', '')
            } for match in fuzzy_matches[:3]]
        
        return []
    
    def _fuzzy_match_hpo(self, phrase: str) -> List[Dict]:
        """
        Perform fuzzy matching between a phrase and HPO terms
        
        Args:
            phrase: The clinical phrase to match
            
        Returns:
            List of matching HPO terms with scores
        """
        matches = []
        
        # Get all HPO terms from the database
        hpo_items = self.hpo_db.hpo_data['items']
        
        # Calculate fuzzy match scores
        for item in hpo_items:
            # Match against the main term
            score = fuzz.token_sort_ratio(phrase.lower(), item['name'].lower())
            
            # Also check synonyms if available
            if 'synonyms' in item:
                for synonym in item['synonyms']:
                    syn_score = fuzz.token_sort_ratio(phrase.lower(), synonym.lower())
                    score = max(score, syn_score)
            
            if score >= self.fuzzy_match_threshold * 0.7:  # Lower threshold for initial filtering
                matches.append({
                    'hpo_id': item['hpo_id'],
                    'name': item['name'],
                    'score': score,
                    'definition': item.get('definition', '')
                })
        
        # Sort by score
        matches.sort(key=lambda x: x['score'], reverse=True)
        
        return matches[:10]  # Return top 10 matches


def parse_hpo_json(json_file: str) -> Dict:
    """
    Parse the HPO JSON file and extract relevant information
    
    Args:
        json_file: Path to the HPO JSON file
        
    Returns:
        Dictionary with HPO terms and metadata
    """
    with open(json_file, 'r') as f:
        hpo_json = json.load(f)
    
    # Extract nodes from the JSON
    nodes = hpo_json.get('graphs', [{}])[0].get('nodes', [])
    
    # Process each node to extract relevant information
    hpo_items = []
    for node in nodes:
        if node.get('type') == 'CLASS':
            item = {
                'hpo_id': node.get('id', '').replace('http://purl.obolibrary.org/obo/HP_', 'HP:'),
                'name': node.get('lbl', '')
            }
            
            # Extract definition if available
            meta = node.get('meta', {})
            if 'definition' in meta:
                item['definition'] = meta['definition'].get('val', '')
                
            # Extract synonyms if available
            if 'synonyms' in meta:
                item['synonyms'] = [syn.get('val', '') for syn in meta['synonyms']]
                
            hpo_items.append(item)
    
    return {'items': hpo_items}


# Main script to run hpomapper
def main():
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description='hpomapper: Extract HPO terms from clinical notes')
    parser.add_argument('--notes', type=str, required=True, help='Path to clinical notes CSV file')
    parser.add_argument('--output', type=str, default='hpomapper_results.csv', help='Path to output CSV file')
    parser.add_argument('--embeddings', type=str, default='G2GHPO_metadata.npy', help='Path to HPO embeddings file')
    parser.add_argument('--model', type=str, default='anthropic.claude-v2', help='Bedrock model ID')
    parser.add_argument('--region', type=str, default=None, help='AWS region')
    parser.add_argument('--profile', type=str, default=None, help='AWS profile name')
    parser.add_argument('--threshold', type=int, default=80, help='Fuzzy matching threshold')
    parser.add_argument('--update-hpo', action='store_true', help='Update HPO embeddings')
    parser.add_argument('--hpo-json', type=str, help='Path to HPO JSON file for updating embeddings')
    
    args = parser.parse_args()
    
    # Use the AWS helper to verify credentials
    aws_session = aws_helper.get_aws_session(
        profile_name=args.profile,
        region_name=args.region
    )
    
    # Initialize components using the same profile/region
    llm_client = BedrockLLM(
        model_id=args.model, 
        region_name=args.region,
        profile_name=args.profile
    )
        
    # If updating HPO embeddings
    if args.update_hpo:
        if not args.hpo_json:
            logger.error("HPO JSON file path required for updating embeddings")
            return
            
        # Parse HPO JSON file
        hpo_data = parse_hpo_json(args.hpo_json)
        
        # TODO: Implement HPO embedding generation
        logger.info("HPO embedding update not fully implemented")
    
    # Initialize HPO vector database
    hpo_db = HPOVectorDB(embedding_file=args.embeddings)
    
    # Initialize hpomapper
    hpomapper = hpomapper(
        llm_client=llm_client,
        hpo_db=hpo_db,
        fuzzy_match_threshold=args.threshold
    )
    
    # Process clinical notes
    results = hpomapper.process_clinical_notes(
        notes_file=args.notes,
        output_file=args.output
    )
    
    logger.info(f"Processed {len(results)} HPO terms")
    print(f"Extracted {len(results)} HPO terms from clinical notes")
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
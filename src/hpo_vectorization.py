import json
import pandas as pd
import numpy as np
import boto3
import time
from pathlib import Path
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("hpo_vectorization.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HPOVectorizer:
    """Class to prepare and vectorize the HPO database"""
    
    def __init__(self, model_id="amazon.titan-embed-text-v2:0", region_name="us-west-2", profile_name="plm-dev"):
        """
        Initialize the HPO Vectorizer
        
        Args:
            model_id: The Bedrock embedding model ID
            region_name: AWS region name
            profile_name: AWS profile name for SSO
        """
        self.model_id = model_id
        
        # Create a boto3 session with the specified profile
        session = boto3.Session(profile_name=profile_name, region_name=region_name)
        
        # Create the bedrock client using the session
        self.bedrock_client = session.client(service_name='bedrock-runtime')
        
        logger.info(f"Initialized Bedrock client with model {model_id}")
    
    def parse_hpo_json(self, json_file):
        """
        Parse the HPO JSON file and extract terms, definitions, and synonyms
        
        Args:
            json_file: Path to the HPO JSON file
            
        Returns:
            DataFrame with HPO terms and metadata
        """
        logger.info(f"Parsing HPO JSON file: {json_file}")
        
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
                else:
                    item['definition'] = ''
                    
                # Extract synonyms if available
                if 'synonyms' in meta:
                    item['synonyms'] = [syn.get('val', '') for syn in meta['synonyms']]
                else:
                    item['synonyms'] = []
                    
                hpo_items.append(item)
        
        # Convert to DataFrame
        df = pd.DataFrame(hpo_items)
        logger.info(f"Extracted {len(df)} HPO terms")
        
        return df
    
    def load_additional_terms(self, addon_file):
        """
        Load additional HPO terms and synonyms from a CSV file
        
        Args:
            addon_file: Path to the CSV file with additional terms
            
        Returns:
            DataFrame with additional terms
        """
        logger.info(f"Loading additional terms from: {addon_file}")
        
        addons_df = pd.read_csv(addon_file)
        
        # Ensure the DataFrame has the required columns
        required_cols = ['hpo_id', 'name', 'definition', 'synonyms']
        for col in required_cols:
            if col not in addons_df.columns:
                if col == 'synonyms':
                    addons_df[col] = addons_df.apply(lambda x: [], axis=1)
                else:
                    addons_df[col] = ''
        
        logger.info(f"Loaded {len(addons_df)} additional terms")
        
        return addons_df
    
    def prepare_hpo_data(self, hpo_df, addons_df=None):
        """
        Prepare HPO data for vectorization
        
        Args:
            hpo_df: DataFrame with HPO terms
            addons_df: DataFrame with additional terms
            
        Returns:
            DataFrame with combined and prepared data
        """
        # Combine main HPO data with addons if provided
        if addons_df is not None:
            combined_df = pd.concat([hpo_df, addons_df], ignore_index=True)
        else:
            combined_df = hpo_df.copy()
        
        # Create text for embedding
        # Include name, definition, and synonyms in the text to embed
        combined_df['text_for_embedding'] = combined_df.apply(
            lambda row: f"{row['name']}. {row['definition']} " + 
                       (". ".join(row['synonyms']) if isinstance(row['synonyms'], list) else ""),
            axis=1
        )
        
        logger.info(f"Prepared {len(combined_df)} HPO terms for vectorization")
        
        return combined_df
    
    def generate_embeddings(self, df, batch_size=20):
        """
        Generate embeddings for HPO terms using Bedrock
        
        Args:
            df: DataFrame with HPO terms and text_for_embedding column
            batch_size: Number of terms to process in each batch
            
        Returns:
            DataFrame with embeddings added
        """
        logger.info("Generating embeddings for HPO terms")
        
        embeddings = []
        
        # Process in batches to avoid API rate limits
        for i in tqdm(range(0, len(df)), desc="Generating embeddings"):
            row = df.iloc[i]
            
            try:
                embedding = self._get_embedding(row['text_for_embedding'])
                embeddings.append(embedding)
                # Small delay to avoid hitting API rate limits
                time.sleep(0.05)
            except Exception as e:
                logger.error(f"Error generating embedding for {row['hpo_id']}: {str(e)}")
                # Add a zero vector as placeholder for failed embeddings
                embeddings.append(np.zeros(1536))  # Assuming 1536-dimensional embeddings
        
        # Add embeddings to DataFrame
        df['embedding'] = embeddings
        
        logger.info(f"Generated embeddings for {len(df)} HPO terms")
        
        return df
    
    def _get_embedding(self, text):
        """
        Get embedding for a single text using Bedrock API
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        try:
            # Simplified format for Titan embed text v2 that worked in our test
            request_body = {
                "inputText": text
            }
            
            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body)
            )
            
            response_body = json.loads(response.get('body').read())
            
            # Extract the embedding
            embedding = np.array(response_body.get('embedding', []))
            
            return embedding
                
        except Exception as e:
            logger.error(f"Bedrock API call failed: {str(e)}")
            raise
    

    
    def save_embeddings(self, df, output_file):
        """
        Save HPO embeddings to a numpy file
        
        Args:
            df: DataFrame with HPO terms and embeddings
            output_file: Path to save the embeddings
        """
        logger.info(f"Saving embeddings to: {output_file}")
        
        # Convert DataFrame to dictionary format
        items = []
        for _, row in df.iterrows():
            item = {
                'hpo_id': row['hpo_id'],
                'name': row['name'],
                'definition': row['definition'],
                'synonyms': row['synonyms'],
                'embedding': row['embedding']
            }
            items.append(item)
        
        # Create the metadata dictionary
        metadata = {
            'items': items,
            'model_id': self.model_id,
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save to numpy file
        np.save(output_file, metadata)
        
        logger.info(f"Saved embeddings for {len(items)} HPO terms")
        
        # Also save a CSV version for inspection (without embeddings)
        csv_file = output_file.replace('.npy', '.csv')
        df_for_csv = df.drop('embedding', axis=1)
        df_for_csv.to_csv(csv_file, index=False)
        
        logger.info(f"Saved CSV version to: {csv_file}")

def create_test_embeddings_file(output_file, num_terms=10, embedding_dim=1536):
    """Create a test embeddings file with mock data for when API access is unavailable"""
    import numpy as np
    import time
    
    print(f"Creating test embeddings file: {output_file}")
    
    # Sample HPO terms
    sample_terms = [
        {"hpo_id": "HP:0001250", "name": "Seizures", "definition": "Sudden disturbances of the brain's normal electrical activity", "synonyms": ["Epileptic seizures", "Fits"]},
        {"hpo_id": "HP:0000252", "name": "Microcephaly", "definition": "Abnormally small head", "synonyms": ["Small head"]},
        {"hpo_id": "HP:0001263", "name": "Developmental delay", "definition": "Delay in achieving developmental milestones", "synonyms": ["Global developmental delay"]},
        {"hpo_id": "HP:0001290", "name": "Hypotonia", "definition": "Decreased muscle tone", "synonyms": ["Muscular hypotonia", "Floppy infant"]},
        {"hpo_id": "HP:0000238", "name": "Hydrocephalus", "definition": "Abnormal accumulation of cerebrospinal fluid", "synonyms": ["Ventriculomegaly"]},
        {"hpo_id": "HP:0012623", "name": "Macrocephaly", "definition": "Abnormally large head", "synonyms": ["Large head"]},
        {"hpo_id": "HP:0004322", "name": "Short stature", "definition": "Height significantly below average", "synonyms": ["Growth retardation"]},
        {"hpo_id": "HP:0000510", "name": "Cataracts", "definition": "Clouding of the lens", "synonyms": ["Lens opacity"]},
        {"hpo_id": "HP:0000365", "name": "Hearing impairment", "definition": "Decreased ability to hear", "synonyms": ["Hearing loss", "Deafness"]},
        {"hpo_id": "HP:0000707", "name": "Abnormality of the nervous system", "definition": "Abnormality of the brain, spinal cord or peripheral nerves", "synonyms": []}
    ]
    
    # Generate random embeddings
    items = []
    for term in sample_terms[:num_terms]:
        # Create a random embedding vector
        embedding = np.random.randn(embedding_dim).astype('float32')
        embedding = embedding / np.linalg.norm(embedding)  # Normalize
        
        # Add the item with the embedding
        term_with_embedding = term.copy()
        term_with_embedding["embedding"] = embedding
        items.append(term_with_embedding)
    
    # Create the metadata dictionary
    metadata = {
        'items': items,
        'model_id': 'mock-embedding-model',
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save to numpy file
    np.save(output_file, metadata)
    print(f"Created test embeddings file with {len(items)} terms at {output_file}")
    
    return metadata

# Main function to run the vectorization process
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='HPO Database Vectorization')
    parser.add_argument('--hpo-json', type=str, required=True, help='Path to the HPO JSON file')
    parser.add_argument('--addons', type=str, help='Path to CSV file with additional HPO terms')
    parser.add_argument('--output', type=str, default='G2GHPO_metadata.npy', help='Path to save the embeddings')
    parser.add_argument('--model', type=str, default='amazon.titan-embed-text-v2:0', help='Bedrock embedding model ID')
    parser.add_argument('--region', type=str, default='us-west-2', help='AWS region')
    parser.add_argument('--profile', type=str, default='plm-dev', help='AWS profile name')
    parser.add_argument('--batch-size', type=int, default=20, help='Batch size for embedding generation')
    parser.add_argument('--test-mode', action='store_true', help='Generate test embeddings without using API')
    
    args = parser.parse_args()
    
    if args.test_mode:
        # Generate test embeddings without API
        create_test_embeddings_file(args.output)
        return
        
    # Initialize the vectorizer
    vectorizer = HPOVectorizer(model_id=args.model, region_name=args.region, profile_name=args.profile)
    
    # Parse HPO JSON
    hpo_df = vectorizer.parse_hpo_json(args.hpo_json)
    
    # Load additional terms if provided
    addons_df = None
    if args.addons:
        addons_df = vectorizer.load_additional_terms(args.addons)
    
    # Prepare data for vectorization
    prepared_df = vectorizer.prepare_hpo_data(hpo_df, addons_df)
    
    # Generate embeddings
    with_embeddings_df = vectorizer.generate_embeddings(prepared_df, batch_size=args.batch_size)
    
    # Save embeddings
    vectorizer.save_embeddings(with_embeddings_df, args.output)
    
    logger.info("HPO vectorization completed successfully")

if __name__ == "__main__":
    main()
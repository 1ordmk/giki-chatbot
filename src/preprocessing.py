"""
Data Preprocessing and Embedding Generation for GIKI Chatbot
Includes: cleaning, chunking, embedding, and vector DB storage
"""

import json
import re
import logging
from pathlib import Path
from typing import List, Dict
import numpy as np

from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / '.env')

import spacy
from sentence_transformers import SentenceTransformer
import pinecone
from sklearn.decomposition import PCA
import umap

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Clean and preprocess scraped text data"""
    
    def __init__(self):
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            logger.warning("Downloading spaCy model...")
            import os
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-]', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        return text.strip()
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        # Process with spaCy for sentence boundaries
        doc = self.nlp(text)
        sentences = [sent.text for sent in doc.sents]
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                # Keep last few sentences for overlap
                overlap_sentences = []
                overlap_length = 0
                for s in reversed(current_chunk):
                    if overlap_length < overlap:
                        overlap_sentences.insert(0, s)
                        overlap_length += len(s.split())
                    else:
                        break
                current_chunk = overlap_sentences
                current_length = overlap_length
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def preprocess_document(self, doc: Dict) -> List[Dict]:
        """Preprocess a single document and create chunks"""
        cleaned_text = self.clean_text(doc['content'])
        
        if len(cleaned_text) < 100:  # Skip very short content
            return []
        
        chunks = self.chunk_text(cleaned_text)
        
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            processed_chunks.append({
                'text': chunk,
                'metadata': {
                    'source_url': doc.get('url', 'unknown'),
                    'title': doc.get('title', 'Unknown'),
                    'category': doc.get('category', 'general'),
                    'chunk_id': i,
                    'total_chunks': len(chunks)
                }
            })
        
        return processed_chunks
    
    def process_dataset(self, input_file: str, output_file: str):
        """Process entire dataset"""
        with open(input_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        all_chunks = []
        for doc in raw_data:
            chunks = self.preprocess_document(doc)
            all_chunks.extend(chunks)
        
        # Save processed data
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Processed {len(raw_data)} documents into {len(all_chunks)} chunks")
        return all_chunks


class EmbeddingGenerator:
    """Generate embeddings using Sentence Transformers"""
    
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = 1024  # Dimension for BAAI/bge-large-en-v1.5
        logger.info(f"Loaded embedding model: {model_name}")
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for list of texts"""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings
    
    def embed_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Add embeddings to chunk dictionaries"""
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.generate_embeddings(texts)
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk['embedding'] = embedding.tolist()
        
        return chunks


class VectorDatabase:
    """Manage Pinecone vector database"""
    
    def __init__(self, api_key: str):
        # Initialize Pinecone with new API
        from pinecone import Pinecone
        self.pc = Pinecone(api_key=api_key)
        self.index_name = "llama-text-embed-v2-index"
        self.dimension = 1024  # Matches the llama-text-embed-v2 model dimensions
        
    def create_index(self):
        """Create Pinecone index if it doesn't exist"""
        try:
            # Check if index exists
            if self.index_name not in self.pc.list_indexes().names():
                # Create new index
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec={
                        "cloud": os.getenv('PINECONE_CLOUD', 'aws'),
                        "region": os.getenv('PINECONE_REGION', 'us-east-1')
                    }
                )
                logger.info(f"Created new index: {self.index_name}")
            
            # Connect to the index
            host = os.getenv('PINECONE_INDEX_HOST')
            if not host:
                raise ValueError("PINECONE_INDEX_HOST environment variable is required")
            
            self.index = self.pc.Index(
                name=self.index_name,
                host=host
            )
            logger.info(f"Connected to index: {self.index_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create/connect to index: {e}")
            return False
    
    def upsert_chunks(self, chunks: List[Dict], batch_size: int = 100):
        """Upload chunks to Pinecone"""
        vectors = []
        
        for i, chunk in enumerate(chunks):
            vector_data = {
                'id': f"chunk_{i}",
                'values': chunk['embedding'],
                'metadata': {
                    'text': chunk['text'][:1000],  # Truncate for metadata limit
                    'source_url': chunk['metadata']['source_url'],
                    'title': chunk['metadata']['title'],
                    'category': chunk['metadata']['category']
                }
            }
            vectors.append(vector_data)
            
            # Upload in batches
            if len(vectors) >= batch_size:
                self.index.upsert(vectors=vectors)
                logger.info(f"Uploaded batch of {len(vectors)} vectors")
                vectors = []
        
        # Upload remaining
        if vectors:
            self.index.upsert(vectors=vectors)
            logger.info(f"Uploaded final batch of {len(vectors)} vectors")
        
        logger.info(f"Total vectors in index: {self.index.describe_index_stats()}")
    
    def query(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """Query similar vectors"""
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        return results.matches
class DimensionalityAnalyzer:
    """Analyze embeddings using PCA and UMAP"""
    
    def __init__(self, embeddings: np.ndarray):
        self.embeddings = embeddings
        self.n_samples = embeddings.shape[0]
        self.n_features = embeddings.shape[1]
        
    def apply_pca(self, n_components: int = 50):
        """Apply PCA dimensionality reduction"""
        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(self.embeddings)
        
        variance_explained = pca.explained_variance_ratio_.sum()
        logger.info(f"PCA: {n_components} components explain {variance_explained:.2%} variance")
        
        return reduced, pca
    
    def apply_umap(self, n_components: int = 2, n_neighbors: int = 15):
        """Apply UMAP for visualization"""
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            metric='cosine'
        )
        reduced = reducer.fit_transform(self.embeddings)
        
        logger.info(f"UMAP: Reduced to {n_components} dimensions")
        return reduced
    
    def save_analysis(self, output_dir: str):
        """Save PCA and UMAP results"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # PCA - use min(n_samples-1, n_features) components
        n_components = min(self.n_samples - 1, self.n_features)
        pca_reduced, pca_model = self.apply_pca(n_components)
        np.save(output_path / f"embeddings_pca{n_components}.npy", pca_reduced)
        
        # UMAP 2D
        umap_2d = self.apply_umap(n_components=2)
        np.save(output_path / "embeddings_umap2d.npy", umap_2d)
        
        # UMAP 3D
        umap_3d = self.apply_umap(n_components=3)
        np.save(output_path / "embeddings_umap3d.npy", umap_3d)
        
        logger.info(f"Saved dimensionality reduction results to {output_path}")


def main():
    """Main pipeline execution"""
    # Step 1: Preprocess data
    logger.info("Step 1: Preprocessing data...")
    preprocessor = DataPreprocessor()
    chunks = preprocessor.process_dataset(
        input_file="data/raw/giki_targeted.json",
        output_file="data/processed/chunks.json"
    )

    # Step 2: Generate embeddings
    logger.info("Step 2: Generating embeddings...")
    embedder = EmbeddingGenerator()
    chunks_with_embeddings = embedder.embed_chunks(chunks)

    # Save chunks with embeddings
    with open("data/processed/chunks_embedded.json", 'w') as f:
        json.dump(chunks_with_embeddings, f, indent=2)

    # Step 3: Dimensionality analysis
    logger.info("Step 3: Performing dimensionality analysis...")
    embeddings_array = np.array([c['embedding'] for c in chunks_with_embeddings])
    analyzer = DimensionalityAnalyzer(embeddings_array)
    analyzer.save_analysis("data/analysis")

    # Step 4: Upload to Pinecone
    logger.info("Step 4: Uploading to Pinecone...")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")

    if pinecone_api_key:
        vector_db = VectorDatabase(api_key=pinecone_api_key)
        ok = vector_db.create_index()
        if ok and getattr(vector_db, 'index', None):
            vector_db.upsert_chunks(chunks_with_embeddings)
            logger.info("Pipeline completed successfully and vectors uploaded to Pinecone!")
        else:
            logger.warning("Pinecone index not available. Vectors were not uploaded.")
            logger.info("If you have a Pinecone project, create an index named 'giki-chatbot' in the Pinecone console, or set PINECONE_CLOUD and PINECONE_REGION environment variables and re-run.")
    else:
        logger.warning("PINECONE_API_KEY not found. Skipping vector DB upload.")
        logger.info("Set environment variable: export PINECONE_API_KEY='your-key'")

if __name__ == "__main__":
    main()
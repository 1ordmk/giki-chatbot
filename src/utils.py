"""
Utility functions and visualization tools for GIKI Chatbot
Includes: mock data generation, embedding visualization, evaluation metrics
"""

import json
import random
import logging
from pathlib import Path
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ====================================================
#  MOCK DATA GENERATION
# ====================================================

class MockDataGenerator:
    """Generate mock student data for admin features"""
    
    def __init__(self):
        self.departments = [
            'Computer Science', 'Electrical Engineering', 'Mechanical Engineering',
            'Chemical Engineering', 'Materials Engineering', 'Management Sciences'
        ]
        self.first_names = [
            'Ahmed', 'Ali', 'Hassan', 'Usman', 'Bilal', 'Hamza',
            'Fatima', 'Ayesha', 'Sara', 'Zainab', 'Maryam', 'Hira'
        ]
        self.last_names = [
            'Khan', 'Ali', 'Ahmed', 'Shah', 'Malik', 'Hussain',
            'Iqbal', 'Raza', 'Haider', 'Abbasi'
        ]
    
    def generate_students(self, n: int = 100) -> Dict:
        """Generate n mock students"""
        students = {}
        
        for i in range(n):
            year = random.choice([2021, 2022, 2023, 2024])
            student_id = f"{year}{random.randint(100, 999)}"
            
            students[student_id] = {
                'name': f"{random.choice(self.first_names)} {random.choice(self.last_names)}",
                'department': random.choice(self.departments),
                'semester': random.randint(1, 8),
                'gpa': round(random.uniform(2.0, 4.0), 2),
                'attendance': round(random.uniform(70.0, 100.0), 1),
                'email': f"{student_id}@giki.edu.pk",
                'status': random.choice(['Active', 'Active', 'Active', 'On Leave'])
            }
        
        return students
    
    def save_mock_data(self, filepath: str = 'data/mock_students.json', n: int = 100):
        """Generate and save mock student data"""
        students = self.generate_students(n)
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(students, f, indent=2)
        
        logger.info(f"Generated {len(students)} mock students in {filepath}")
        return students


# ====================================================
#  EMBEDDING VISUALIZATION
# ====================================================

class EmbeddingVisualizer:
    """Visualize embeddings using PCA and UMAP"""
    
    def __init__(self, embeddings_path: str = None, chunks_path: str = None):
        if embeddings_path and Path(embeddings_path).exists():
            with open(chunks_path, 'r') as f:
                chunks = json.load(f)
            self.embeddings = np.array([c['embedding'] for c in chunks])
            self.metadata = [c['metadata'] for c in chunks]
        else:
            self.embeddings = None
            self.metadata = None
    
    def plot_umap_2d(self, output_path: str = 'data/analysis/umap_visualization.png'):
        """Create 2D UMAP visualization"""
        if self.embeddings is None:
            logger.error("No embeddings loaded!")
            return
        
        umap_2d_path = 'data/analysis/embeddings_umap2d.npy'
        if not Path(umap_2d_path).exists():
            logger.error(f"{umap_2d_path} not found. Run preprocessing first.")
            return
        
        umap_coords = np.load(umap_2d_path)
        
        categories = [m.get('category', 'general') for m in self.metadata]
        unique_categories = list(set(categories))
        color_map = {cat: plt.cm.tab10(i) for i, cat in enumerate(unique_categories)}
        
        plt.figure(figsize=(12, 8))
        for cat in unique_categories:
            mask = np.array(categories) == cat
            plt.scatter(
                umap_coords[mask, 0],
                umap_coords[mask, 1],
                c=[color_map[cat]],
                label=cat,
                alpha=0.6,
                s=50
            )
        
        plt.title('UMAP Projection of Document Embeddings', fontsize=16, fontweight='bold')
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved UMAP visualization to {output_path}")
    
    def plot_pca_variance(self, output_path: str = 'data/analysis/pca_variance.png'):
        """Plot PCA explained variance"""
        from sklearn.decomposition import PCA
        
        if self.embeddings is None:
            logger.error("No embeddings loaded!")
            return
        
        pca = PCA()
        pca.fit(self.embeddings)
        
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(cumsum) + 1), cumsum, 'b-', linewidth=2)
        plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
        plt.axhline(y=0.90, color='g', linestyle='--', label='90% Variance')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('PCA Explained Variance Analysis')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved PCA variance plot to {output_path}")
    
    def find_similar_chunks(self, query_embedding: np.ndarray, top_k: int = 5):
        """Find most similar chunks to a query"""
        if self.embeddings is None:
            logger.error("No embeddings loaded!")
            return []
        
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'similarity': float(similarities[idx]),
                'metadata': self.metadata[idx]
            })
        
        return results


# ====================================================
#  CHATBOT EVALUATION
# ====================================================

class ChatbotEvaluator:
    """Evaluate chatbot performance"""
    
    def __init__(self):
        self.test_queries = self._load_test_queries()
    
    def _load_test_queries(self) -> List[Dict]:
        """Load or create test queries with expected answers"""
        return [
            {
                'query': 'What programs does GIKI offer?',
                'expected_keywords': ['engineering', 'computer', 'management']
            },
            {
                'query': 'Who is the rector of GIKI?',
                'expected_keywords': ['rector', 'GIKI']
            },
            {
                'query': 'What is the GPA requirement for graduation?',
                'expected_keywords': ['GPA', 'requirement']
            },
            {
                'query': 'Where is the student hostel located?',
                'expected_keywords': ['hostel', 'location']
            },
            {
                'query': 'How can I contact admissions office?',
                'expected_keywords': ['admissions', 'contact', 'email']
            }
        ]
    
    def evaluate_response(self, response: str, expected_keywords: List[str]) -> float:
        """Compute keyword recall accuracy"""
        response = response.lower()
        matched = sum(1 for kw in expected_keywords if kw.lower() in response)
        return matched / len(expected_keywords)
    
    def evaluate_chatbot(self, chatbot_func) -> Dict:
        """
        Evaluate chatbot on test queries.
        chatbot_func: function(query) -> response
        """
        scores = []
        for q in self.test_queries:
            response = chatbot_func(q['query'])
            accuracy = self.evaluate_response(response, q['expected_keywords'])
            scores.append(accuracy)
            logger.info(f"Query: {q['query']} | Score: {accuracy:.2f}")
        
        mean_score = np.mean(scores)
        logger.info(f"Average chatbot keyword recall: {mean_score:.2f}")
        return {"average_score": mean_score, "individual_scores": scores}


# ====================================================
#  MAIN TEST
# ====================================================

if __name__ == "__main__":
    # Generate mock data for testing
    generator = MockDataGenerator()
    generator.save_mock_data(n=50)

    # Evaluate with dummy chatbot
    evaluator = ChatbotEvaluator()
    dummy_chatbot = lambda q: "GIKI offers various engineering programs and management sciences."
    evaluator.evaluate_chatbot(dummy_chatbot)

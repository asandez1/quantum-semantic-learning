"""
Data Preparation Pipeline for Q-Manifold (Paper 4)

Prepares ConceptNet semantic pairs for quantum metric refinement:
1. Load concept embeddings (from sentence-transformers)
2. Apply PCA reduction to 20D (Paper 1 intrinsic dimensionality)
3. Scale to [0, π] range for angle encoding
4. Compute hyperbolic distances for contrastive loss targets
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict, Optional
import pickle
import os


class QManifoldDataPreparation:
    """
    Prepares semantic data for quantum metric refinement.

    Pipeline:
    - High-D embeddings (384D/768D) → PCA → 20D → MinMax scaling → [0.1, π-0.1]
    """

    def __init__(
        self,
        model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
        target_dim: int = 20,
        cache_dir: str = '../data'
    ):
        """
        Args:
            model_name: Sentence transformer model
            target_dim: Target dimensionality (default: 20 from Paper 1)
            cache_dir: Directory for caching PCA and embeddings
        """
        print(f"[Data Prep] Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.target_dim = target_dim
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        # Initialize PCA and scaler (will be fitted on data)
        self.pca = PCA(n_components=target_dim)
        self.scaler = MinMaxScaler(feature_range=(0.1, np.pi - 0.1))

        self.is_fitted = False

        print(f"[Data Prep] Embedding dim: {self.embedding_dim} → Target: {target_dim}D")

    def get_default_concept_pairs(self) -> List[Tuple[str, str]]:
        """
        Returns a curated list of semantic concept pairs from ConceptNet.
        These pairs have clear hierarchical relationships (hyperbolic geometry).

        Returns:
            List of (concept1, concept2) tuples
        """
        # Hierarchical pairs (hypernym/hyponym relationships)
        hierarchical_pairs = [
            ("animal", "mammal"),
            ("mammal", "dog"),
            ("dog", "poodle"),
            ("vehicle", "car"),
            ("car", "sedan"),
            ("fruit", "apple"),
            ("apple", "granny_smith"),
            ("building", "house"),
            ("house", "cottage"),
            ("emotion", "happiness"),
            ("happiness", "joy"),
            ("color", "blue"),
            ("blue", "navy"),
            ("instrument", "guitar"),
            ("guitar", "acoustic_guitar"),
            ("science", "physics"),
            ("physics", "quantum_mechanics"),
            ("sport", "football"),
            ("football", "soccer"),
            ("food", "dessert"),
            ("dessert", "cake"),
            ("cake", "chocolate_cake"),
            ("tool", "hammer"),
            ("profession", "doctor"),
            ("doctor", "surgeon"),
            ("plant", "tree"),
            ("tree", "oak"),
            ("beverage", "coffee"),
            ("coffee", "espresso"),
            ("furniture", "chair"),
            ("chair", "armchair"),
            ("weather", "storm"),
            ("storm", "hurricane"),
            ("shape", "polygon"),
            ("polygon", "triangle"),
            ("material", "metal"),
            ("metal", "iron"),
            ("art", "painting"),
            ("painting", "portrait"),
            ("language", "programming"),
            ("programming", "python"),
            ("mathematics", "algebra"),
            ("algebra", "linear_algebra"),
            ("energy", "electricity"),
            ("electricity", "current"),
            ("literature", "poetry"),
            ("poetry", "sonnet"),
            ("music", "jazz"),
            ("jazz", "bebop"),
            ("dance", "ballet"),
            ("ballet", "classical_ballet"),
        ]

        return hierarchical_pairs[:50]  # Use 50 pairs for probe

    def generate_all_concepts(self, pairs: List[Tuple[str, str]]) -> List[str]:
        """Extract unique concepts from pairs."""
        concepts = set()
        for c1, c2 in pairs:
            concepts.add(c1)
            concepts.add(c2)
        return sorted(list(concepts))

    def embed_concepts(
        self,
        concepts: List[str],
        use_cache: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for concepts.

        Args:
            concepts: List of concept strings
            use_cache: Whether to use cached embeddings

        Returns:
            Array of embeddings (num_concepts × embedding_dim)
        """
        cache_file = os.path.join(self.cache_dir, 'concept_embeddings.pkl')

        # Try cache
        if use_cache and os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                cache = pickle.load(f)

            if all(c in cache for c in concepts):
                print(f"[Data Prep] Loaded {len(concepts)} embeddings from cache")
                return np.array([cache[c] for c in concepts])

        # Generate embeddings
        print(f"[Data Prep] Generating embeddings for {len(concepts)} concepts...")
        embeddings = self.model.encode(concepts, show_progress_bar=True)

        # Update cache
        if use_cache:
            cache = {}
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    cache = pickle.load(f)

            for concept, embedding in zip(concepts, embeddings):
                cache[concept] = embedding

            with open(cache_file, 'wb') as f:
                pickle.dump(cache, f)

            print(f"[Data Prep] Cache updated: {len(cache)} total embeddings")

        return embeddings

    def fit_transform_pca(
        self,
        embeddings: np.ndarray,
        save: bool = True
    ) -> np.ndarray:
        """
        Fit PCA and reduce to target dimensionality.

        Args:
            embeddings: High-dimensional embeddings
            save: Whether to save the PCA model

        Returns:
            Reduced embeddings (num_concepts × target_dim)
        """
        print(f"[Data Prep] Fitting PCA: {embeddings.shape[1]}D → {self.target_dim}D")

        # Fit and transform
        reduced = self.pca.fit_transform(embeddings)

        # Fit scaler on reduced data
        scaled = self.scaler.fit_transform(reduced)

        self.is_fitted = True

        # Print variance explained
        var_explained = np.sum(self.pca.explained_variance_ratio_)
        print(f"[Data Prep] Variance explained: {var_explained:.3f}")
        print(f"[Data Prep] Scaled range: [{scaled.min():.3f}, {scaled.max():.3f}]")

        # Save models
        if save:
            pca_file = os.path.join(self.cache_dir, 'pca_model.pkl')
            scaler_file = os.path.join(self.cache_dir, 'scaler_model.pkl')

            with open(pca_file, 'wb') as f:
                pickle.dump(self.pca, f)
            with open(scaler_file, 'wb') as f:
                pickle.dump(self.scaler, f)

            print(f"[Data Prep] Saved PCA and scaler models")

        return scaled

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply fitted PCA + scaling."""
        if not self.is_fitted:
            raise ValueError("Must call fit_transform_pca first!")

        reduced = self.pca.transform(embeddings)
        scaled = self.scaler.transform(reduced)
        return scaled

    def compute_hyperbolic_distance(
        self,
        u: np.ndarray,
        v: np.ndarray,
        c: float = 1.0
    ) -> float:
        """
        Compute hyperbolic distance in Poincaré disk model.

        Args:
            u, v: Points in Euclidean space (will be projected to disk)
            c: Curvature parameter

        Returns:
            Hyperbolic distance
        """
        # Normalize to unit ball
        u_norm = np.linalg.norm(u)
        v_norm = np.linalg.norm(v)

        if u_norm >= 1.0:
            u = u / (u_norm + 1e-3)
        if v_norm >= 1.0:
            v = v / (v_norm + 1e-3)

        # Hyperbolic distance formula
        diff_squared = np.sum((u - v)**2)
        denom = (1 - np.sum(u**2)) * (1 - np.sum(v**2))

        if denom <= 1e-10:
            return 0.0

        delta = 2.0 * diff_squared / (denom + 1e-10)
        distance = np.arccosh(1.0 + delta)

        return distance * np.sqrt(c)

    def hyperbolic_similarity(self, distance: float, scale: float = 1.0) -> float:
        """
        Convert hyperbolic distance to similarity [0, 1].

        Args:
            distance: Hyperbolic distance
            scale: Scaling factor

        Returns:
            Similarity score
        """
        return np.exp(-distance / scale)

    def prepare_training_batch(
        self,
        pairs: List[Tuple[str, str]],
        batch_size: int = 8
    ) -> Dict:
        """
        Prepare a complete training batch.

        Args:
            pairs: List of concept pairs
            batch_size: Number of pairs per batch

        Returns:
            Dictionary with:
            - 'vectors_20d': Scaled 20D vectors for all concepts
            - 'concept_to_idx': Mapping from concept name to index
            - 'pair_indices': List of (idx1, idx2) for each pair
            - 'target_similarities': Target similarity scores
            - 'batches': List of batch dictionaries
        """
        # Get all unique concepts
        concepts = self.generate_all_concepts(pairs)

        # Generate embeddings
        embeddings = self.embed_concepts(concepts)

        # PCA reduction (but don't scale yet - need for hyperbolic distances)
        print(f"[Data Prep] Fitting PCA: {embeddings.shape[1]}D → {self.target_dim}D")
        vectors_pca = self.pca.fit_transform(embeddings)

        # Print variance explained
        var_explained = np.sum(self.pca.explained_variance_ratio_)
        print(f"[Data Prep] Variance explained: {var_explained:.3f}")

        # Create concept-to-index mapping
        concept_to_idx = {c: i for i, c in enumerate(concepts)}

        # CRITICAL FIX: Compute hyperbolic distances on PCA vectors (BEFORE scaling)
        # These vectors need to be in Poincaré disk for proper hyperbolic distance computation
        pair_indices = []
        target_similarities = []

        for c1, c2 in pairs:
            idx1 = concept_to_idx[c1]
            idx2 = concept_to_idx[c2]
            pair_indices.append((idx1, idx2))

            # Compute hyperbolic distance on UNSCALED PCA vectors
            v1_pca = vectors_pca[idx1]
            v2_pca = vectors_pca[idx2]
            dist = self.compute_hyperbolic_distance(v1_pca, v2_pca)
            sim = self.hyperbolic_similarity(dist)
            target_similarities.append(sim)

        # NOW apply scaling for angle encoding (for quantum circuits)
        vectors_20d = self.scaler.fit_transform(vectors_pca)
        self.is_fitted = True

        print(f"[Data Prep] Scaled range: [{vectors_20d.min():.3f}, {vectors_20d.max():.3f}]")

        # Save models
        pca_file = os.path.join(self.cache_dir, 'pca_model.pkl')
        scaler_file = os.path.join(self.cache_dir, 'scaler_model.pkl')
        with open(pca_file, 'wb') as f:
            pickle.dump(self.pca, f)
        with open(scaler_file, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"[Data Prep] Saved PCA and scaler models")

        # Create batches
        batches = []
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pair_indices[i:i + batch_size]
            batch_targets = target_similarities[i:i + batch_size]

            batches.append({
                'pair_indices': batch_pairs,
                'target_similarities': batch_targets
            })

        print(f"[Data Prep] Prepared {len(pairs)} pairs → {len(batches)} batches")
        print(f"[Data Prep] Target similarity range: [{min(target_similarities):.3f}, {max(target_similarities):.3f}]")

        return {
            'vectors_20d': vectors_20d,
            'concept_to_idx': concept_to_idx,
            'concepts': concepts,
            'pair_indices': pair_indices,
            'target_similarities': target_similarities,
            'batches': batches
        }


if __name__ == "__main__":
    # Test the data preparation pipeline
    print("=" * 60)
    print("Testing Q-Manifold Data Preparation")
    print("=" * 60)

    prep = QManifoldDataPreparation(target_dim=20)
    pairs = prep.get_default_concept_pairs()

    print(f"\nUsing {len(pairs)} concept pairs")
    print(f"Example pairs: {pairs[:3]}")

    # Prepare training batch
    data = prep.prepare_training_batch(pairs, batch_size=8)

    print(f"\nResults:")
    print(f"  - Total concepts: {len(data['concepts'])}")
    print(f"  - Vector shape: {data['vectors_20d'].shape}")
    print(f"  - Number of batches: {len(data['batches'])}")
    print(f"  - Batch 0 size: {len(data['batches'][0]['pair_indices'])}")

    print("\n✓ Data preparation pipeline working!")

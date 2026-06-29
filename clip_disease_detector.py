# ============================================
# clip_disease_detector.py
# Use CLIP to match images to disease descriptions
# NO TRAINING - Uses pre-trained CLIP model
# ============================================

import torch
import clip
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from typing import List, Dict, Tuple


class CLIPDiseaseDetector:
    """
    Zero-shot disease detection using CLIP
    
    How it works:
    1. Convert image to embedding
    2. Convert disease descriptions to embeddings
    3. Find closest matching description
    4. That description = detected disease
    
    NO TRAINING NEEDED!
    Uses OpenAI's CLIP model (free, open source)
    """

    # Disease descriptions for CLIP matching
    DISEASE_DESCRIPTIONS = {

        # Apple diseases
        "Apple - Apple Scab": [
            "apple leaf with dark olive green scab spots",
            "apple leaf showing dark brown crusty lesions",
            "apple leaf with velvety brown circular spots",
        ],
        "Apple - Black Rot": [
            "apple leaf with brown circular spots and rings",
            "apple leaf showing frogeye leaf spot pattern",
            "apple leaf with brown rotting circular lesions",
        ],
        "Apple - Cedar Apple Rust": [
            "apple leaf with bright orange yellow spots",
            "apple leaf with rust colored circular spots",
            "apple leaf showing orange fungal spots",
        ],
        "Apple - Healthy": [
            "healthy green apple leaf no disease",
            "fresh clean apple leaf uniform green color",
        ],

        # Tomato diseases
        "Tomato - Late Blight": [
            "tomato leaf with large dark brown blotches",
            "tomato leaf showing water soaked brown lesions",
            "tomato leaf with white fuzzy mold on back",
            "tomato leaf with dark irregular spreading spots",
        ],
        "Tomato - Early Blight": [
            "tomato leaf with brown spots and yellow rings",
            "tomato leaf showing target board circular spots",
            "tomato leaf with concentric ring brown lesions",
        ],
        "Tomato - Leaf Mold": [
            "tomato leaf with yellow patches and mold",
            "tomato leaf with olive green mold underneath",
            "tomato leaf showing yellowing with fungal mold",
        ],
        "Tomato - Septoria Leaf Spot": [
            "tomato leaf with many small circular spots",
            "tomato leaf with dark bordered circular lesions",
            "tomato leaf showing numerous small brown spots",
        ],
        "Tomato - Bacterial Spot": [
            "tomato leaf with small dark water soaked spots",
            "tomato leaf with angular brown spots",
            "tomato leaf showing bacterial lesions with halos",
        ],
        "Tomato - Healthy": [
            "healthy green tomato leaf no disease",
            "fresh tomato leaf uniform bright green",
        ],

        # Potato diseases
        "Potato - Early Blight": [
            "potato leaf with brown circular target spots",
            "potato leaf showing concentric ring lesions",
            "potato leaf with dark brown spots yellow halos",
        ],
        "Potato - Late Blight": [
            "potato leaf with dark water soaked patches",
            "potato leaf showing gray green brown lesions",
            "potato leaf with white mold on underside",
        ],
        "Potato - Healthy": [
            "healthy potato leaf dark green no spots",
            "fresh potato plant leaf no disease",
        ],

        # Corn diseases
        "Corn - Northern Leaf Blight": [
            "corn leaf with long gray tan cigar shaped lesions",
            "corn leaf showing elongated grayish brown spots",
            "corn leaf with large tan elliptical lesions",
        ],
        "Corn - Common Rust": [
            "corn leaf with small oval rust colored pustules",
            "corn leaf showing reddish brown rust spots",
            "corn leaf with scattered orange brown rust",
        ],
        "Corn - Cercospora Leaf Spot": [
            "corn leaf with rectangular gray lesions",
            "corn leaf showing narrow elongated gray spots",
        ],
        "Corn - Healthy": [
            "healthy corn maize leaf bright green",
            "fresh corn leaf no disease uniform green",
        ],

        # Grape diseases
        "Grape - Black Rot": [
            "grape leaf with circular brown spots black dots",
            "grape leaf showing brown lesions with pycnidia",
        ],
        "Grape - Leaf Blight": [
            "grape leaf with irregular brown blight patches",
            "grape leaf showing widespread brown lesions",
        ],
        "Grape - Healthy": [
            "healthy grape vine leaf green no disease",
            "fresh grape leaf uniform green no spots",
        ],

        # General
        "Powdery Mildew": [
            "leaf covered with white powdery coating",
            "leaf showing white dusty fungal growth",
            "leaf with white flour-like powder on surface",
        ],
        "Healthy Plant": [
            "healthy green plant leaf no visible disease",
            "fresh normal leaf no spots no discoloration",
            "vibrant green leaf perfect health condition",
        ],
    }

    def __init__(self, device: str = None, verbose: bool = True):
        """Initialize CLIP model"""

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.verbose = verbose

        if self.verbose:
            print("⏳ Loading CLIP model...")
            print("   (Downloads ~350MB first time)")

        # Load CLIP (ViT-B/32 is fastest, ViT-L/14 is best)
        self.model, self.preprocess = clip.load(
            "ViT-B/32",
            device=self.device
        )

        if self.verbose:
            print(f"✅ CLIP loaded on: {self.device}")
            print(f"   Diseases in database: {len(self.DISEASE_DESCRIPTIONS)}")

        # Pre-compute text embeddings
        self._precompute_text_embeddings()

    def _precompute_text_embeddings(self):
        """Pre-compute all disease text embeddings"""

        if self.verbose:
            print("⏳ Computing disease embeddings...")

        self.disease_names    = []
        self.text_embeddings  = []

        all_descs = []
        desc_to_disease = []
        for disease, descriptions in self.DISEASE_DESCRIPTIONS.items():
            for desc in descriptions:
                all_descs.append(desc)
                desc_to_disease.append(disease)

        # Batch tokenize all descriptions
        texts = clip.tokenize(all_descs).to(self.device)
        
        with torch.no_grad():
            embeds = self.model.encode_text(texts)
            embeds = embeds / embeds.norm(dim=-1, keepdim=True)
            embeds = embeds.cpu().numpy()

        # Group by disease and average
        disease_to_embeds = {disease: [] for disease in self.DISEASE_DESCRIPTIONS.keys()}
        for embed, disease in zip(embeds, desc_to_disease):
            disease_to_embeds[disease].append(embed)

        for disease, embeds_list in disease_to_embeds.items():
            avg_embed = np.mean(embeds_list, axis=0)
            # Normalize the average embedding
            avg_embed = avg_embed / np.linalg.norm(avg_embed)
            self.disease_names.append(disease)
            self.text_embeddings.append(avg_embed)

        self.text_embeddings = np.vstack(self.text_embeddings)
        if self.verbose:
            print(f"✅ {len(self.disease_names)} disease embeddings ready")

    def detect(
        self,
        image_path: str,
        top_k: int = 5
    ) -> Dict:
        """
        Detect disease in image using CLIP
        
        Returns top-k matching diseases with scores
        """

        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_input = self.preprocess(image).unsqueeze(0)
        image_input = image_input.to(self.device)

        # Get image embedding
        with torch.no_grad():
            image_embed = self.model.encode_image(image_input)
            image_embed = image_embed / image_embed.norm(
                dim=-1, keepdim=True
            )
            image_embed = image_embed.cpu().numpy()

        # Compute similarities
        similarities = np.dot(
            image_embed, self.text_embeddings.T
        )[0]

        # Get top-k results
        top_k_idx = np.argsort(similarities)[::-1][:top_k]

        results = {
            "top_prediction": {
                "disease"   : self.disease_names[top_k_idx[0]],
                "confidence": float(similarities[top_k_idx[0]]) * 100,
                "is_healthy": "healthy" in
                    self.disease_names[top_k_idx[0]].lower()
            },
            "top_k_predictions": [
                {
                    "rank"      : i + 1,
                    "disease"   : self.disease_names[idx],
                    "score"     : float(similarities[idx]),
                    "confidence": float(similarities[idx]) * 100
                }
                for i, idx in enumerate(top_k_idx)
            ]
        }

        return results

    def add_custom_disease(
        self,
        disease_name: str,
        descriptions: List[str]
    ):
        """Add new disease to detector without retraining"""

        if self.verbose:
            print(f"➕ Adding: {disease_name}")

        disease_embeds = []
        for desc in descriptions:
            text = clip.tokenize([desc]).to(self.device)
            with torch.no_grad():
                embed = self.model.encode_text(text)
                embed = embed / embed.norm(
                    dim=-1, keepdim=True
                )
                disease_embeds.append(embed.cpu().numpy())

        avg_embed = np.mean(disease_embeds, axis=0)
        self.disease_names.append(disease_name)
        self.text_embeddings = np.vstack([
            self.text_embeddings,
            avg_embed
        ])

        if self.verbose:
            print(f"✅ {disease_name} added!")
            print(
                f"   Total diseases: {len(self.disease_names)}"
            )


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) > 1 and "--image" in sys.argv:
        try:
            image_idx = sys.argv.index("--image") + 1
            if image_idx >= len(sys.argv):
                print(json.dumps({"success": False, "error": "No image path specified after --image"}))
                sys.exit(1)
            image_path = sys.argv[image_idx]

            detector = CLIPDiseaseDetector(verbose=False)
            result = detector.detect(image_path, top_k=5)
            result["success"] = True
            print(json.dumps(result))
        except Exception as e:
            print(json.dumps({"success": False, "error": str(e)}))
    else:
        # Default behavior: run on test_diseased_leaf.png
        detector = CLIPDiseaseDetector(verbose=True)

        # Detect disease
        result = detector.detect("test_diseased_leaf.png", top_k=5)

        print("\n🌿 CLIP DETECTION RESULTS:")
        print("=" * 50)
        print(f"Top Disease: {result['top_prediction']['disease']}")
        print(f"Confidence : {result['top_prediction']['confidence']:.1f}%")
        print(f"Healthy    : {result['top_prediction']['is_healthy']}")

        print("\n📊 Top 5 Matches:")
        for pred in result['top_k_predictions']:
            print(
                f"  {pred['rank']}. {pred['disease'][:40]:40s} "
                f"→ {pred['confidence']:.1f}%"
            )

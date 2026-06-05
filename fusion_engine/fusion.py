import numpy as np

class MultimodalFusionEngine:
    """Combines predictions from multiple modules using weighted averaging."""
    def __init__(self, voice_weight=0.45, handwriting_weight=0.35, eye_weight=0.20):
        self.default_weights = {
            'voice': voice_weight,
            'handwriting': handwriting_weight,
            'eye': eye_weight
        }

    def fuse(self, predictions):
        """
        Fuse probabilities from available modules.
        :param predictions: Dict with module names as keys and probabilities as values.
        """
        available_modules = []
        cleaned_predictions = {}

        for module_name, value in predictions.items():
            if module_name not in self.default_weights or value is None:
                continue

            numeric_value = float(value)
            cleaned_predictions[module_name] = min(max(numeric_value, 0.0), 1.0)
            available_modules.append(module_name)

        if not available_modules:
            return None
        
        # Calculate current total weight of available modules
        total_weight = sum([self.default_weights[m] for m in available_modules])
        
        # Normalize weights so they sum to 1
        normalized_weights = {m: self.default_weights[m] / total_weight for m in available_modules}
        
        # Calculate final verdict
        final_probability = 0
        for m in available_modules:
            final_probability += normalized_weights[m] * cleaned_predictions[m]
            
        return final_probability

    def get_clinical_verdict(self, final_probability, threshold=0.5):
        """Determine clinical verdict based on fused probability."""
        if final_probability is None:
            return "Inconclusive"
        return "Parkinson's Disease Likely" if final_probability >= threshold else "Healthy"

import pytest
from PIL import Image
import requests
from transformers import ViTFeatureExtractor, ViTForImageClassification, BatchFeature
from transformers.modeling_outputs import ImageClassifierOutput


class TestImageClassification:
    @pytest.mark.parametrize("image_url", [
        "http://images.cocodataset.org/val2017/000000039769.jpg",
        "https://images.pexels.com/photos/13458913/pexels-photo-13458913.jpeg",
        "https://images.pexels.com/photos/13578883/pexels-photo-13578883.jpeg",
        "https://images.pexels.com/photos/1878293/pexels-photo-1878293.jpeg",
        "https://images.pexels.com/photos/919273/pexels-photo-919273.jpeg",
        "https://images.pexels.com/photos/9050512/pexels-photo-9050512.jpeg",
        "https://images.pexels.com/photos/7125781/pexels-photo-7125781.jpeg",
    ])
    def test_vision_transformer(self, image_url: str) -> None:
        image: Image = Image.open(requests.get(url=image_url, stream=True).raw)

        feature_extractor: ViTFeatureExtractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
        model: ViTForImageClassification = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

        inputs: BatchFeature = feature_extractor(images=image, return_tensors="pt")
        outputs: ImageClassifierOutput = model(**inputs)

        predicted_class_idx: int = outputs.logits.argmax(-1).item()
        predicted_class: str = model.config.id2label[predicted_class_idx]

        print("Predicted class:", predicted_class)

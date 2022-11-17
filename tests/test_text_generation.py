import datetime
import os
import time
from enum import Enum, unique
from typing import Callable, Dict, List

import pytest
from transformers import Pipeline, pipeline, set_seed


# TODO: Switch to StrEnum when PyTorch supports Python 3.11.
@unique
class Model(str, Enum):
    GPT2_LARGE = "gpt2-large"
    GTP_NEO_2_7B = "EleutherAI/gpt-neo-2.7B"

    def __str__(self) -> str:
        return self.value


@pytest.mark.skip("takes too long")
class TestTextGeneration:
    output_dir: str = "build/test/TestTextGeneration"

    def __int__(self) -> None:
        self.start_time: float = 0.0

    @classmethod
    def setup_class(cls) -> None:
        try:
            os.makedirs(TestTextGeneration.output_dir, exist_ok=True)
        except OSError:
            pass

    def setup_method(self, method: Callable) -> None:
        self.start_time = time.time()
        set_seed(int(self.start_time))

    @pytest.mark.parametrize(
        "name, input_text, model",
        [
            ("local-seo", "why is local SEO important for small business?", Model.GPT2_LARGE),
            ("local-seo", "why is local SEO important for small business?", Model.GTP_NEO_2_7B),
            ("email-marketing", "leveraging email marketing in small business", Model.GPT2_LARGE),
            ("email-marketing", "leveraging email marketing in small business", Model.GTP_NEO_2_7B),
            ("2022-electric-cars", "what's the best electric car to purchase in 2022?", Model.GPT2_LARGE),
            ("2022-electric-cars", "what's the best electric car to purchase in 2022?", Model.GTP_NEO_2_7B),
            ("machine-learning", "why is machine learning so important in modern technology", Model.GPT2_LARGE),
            ("machine-learning", "why is machine learning so important in modern technology", Model.GTP_NEO_2_7B),
        ],
    )
    def test_long_form_text_generation(self, name: str, input_text: str, model: Model) -> None:
        generator: Pipeline = pipeline(task="text-generation", model=model)
        output: List[Dict] = generator(input_text, min_length=100, max_length=500, num_return_sequences=5)

        self._write_output(name=name, model=model, output=output, end_time=time.time())

    def _write_output(self, name: str, model: Model, output: List[Dict], end_time: float) -> None:
        with open(f"{TestTextGeneration.output_dir}/{name}-{model.name}-{int(end_time)}.txt", "a") as file:
            for generated_text in [result["generated_text"] for result in output]:
                file.write(generated_text)
                file.write("\n\n==========\n\n")
            file.write(f">>>> Run time: {str(datetime.timedelta(seconds=end_time - self.start_time))}\n\n\n")

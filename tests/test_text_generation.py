import datetime
import os
import sys
import time

import pytest
from transformers import pipeline, set_seed


class TestTextGeneration:
    output_dir: str = "build/test/TestTextGeneration"

    @classmethod
    def setup_class(cls) -> None:
        try:
            os.makedirs(TestTextGeneration.output_dir, exist_ok=True)
        except OSError:
            pass

    @pytest.mark.parametrize(
        "input_text",
        [
            "why is local SEO important for small business?",
            "leveraging email marketing in small business",
            "what's the best electric car to purchase in 2022?",
            "why is machine learning so important for modern technology",
        ],
    )
    def test_gpt2_large(self, input_text: str) -> None:
        start_time: float = time.time()

        set_seed(int(start_time))

        generator = pipeline(task="text-generation", model="gpt2-large")
        output = generator(text_inputs=input_text, min_length=500, max_length=1000, num_return_sequences=5)

        end_time: float = time.time()

        with open(
            f"{TestTextGeneration.output_dir}/{sys._getframe().f_code.co_name}-{int(end_time)}.json", "a"
        ) as file:
            for generated_text in [result["generated_text"] for result in output]:
                file.write(generated_text)
                file.write("\n\n==========\n\n")
            file.write(f">>>> Run time: {str(datetime.timedelta(seconds=end_time - start_time))}\n\n\n")

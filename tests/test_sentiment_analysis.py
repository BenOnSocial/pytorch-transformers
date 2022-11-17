import json
from typing import Dict, List

import pytest
from transformers import Pipeline, pipeline


class TestSentimentAnalysis:
    @pytest.mark.parametrize(
        "tweet",
        [
            "Maybe just maybe, the Supreme Court should realize they have no business telling anyone who they can and can't love and marry. I was proud to cast my vote today to finally make marriage equality – including same-sex and interracial marriages – the long overdue law of the land.",
            "President Xi Jinping of China accused Canada's prime minister, Justin Trudeau, of leaking details from an earlier conversation between them to reporters. The exchange happened at a reception in Bali, Indonesia, on Wednesday. https://nyti.ms/3TCa80I",
            "Elon Musk could be chilling in a $100,000,000 mansion right now, getting massaged by 100 women, as a team of private chefs prepares him a 5-star meal! Instead @elonmusk is literally sleeping at Twitter HQ and working his butt off daily to improve the world.",
            "President Trump announces his '24 Presidential run. He is a complete and total embarrassment!",
            "NASA’s Artemis is in flight. This ship will enable the first woman and first person of color to set foot on the lunar surface and will lead countless students to become explorers and show America’s limitless possibilities to the world.",
        ],
    )
    def test_tweet_emotion_analysis(self, tweet: str) -> None:
        classifier: Pipeline = pipeline(
            task="sentiment-analysis", model="finiteautomata/bertweet-base-emotion-analysis", return_all_scores=True
        )

        output: List[Dict] = classifier(tweet)

        print(tweet)
        print(json.dumps(output, indent=2))

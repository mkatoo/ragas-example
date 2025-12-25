from datasets import Dataset
from google import genai
from ragas import evaluate
from ragas.llms import llm_factory
from ragas.metrics import Faithfulness


def main() -> None:
    dataset = Dataset.from_dict(
        {
            "question": ["What is the capital of France?"],
            "answer": [
                "The capital of France is Paris.",
            ],
            "contexts": [
                [
                    "France's capital city is Paris, known for its art, fashion, and culture.",
                    "Berlin is the capital of Germany.",
                ],
            ],
        }
    )
    client = genai.Client()
    llm = llm_factory(model="gemini-2.5-flash-lite", provider="google", client=client)
    results = evaluate(
        dataset,
        metrics=[
            Faithfulness(llm=llm),
        ],
    )

    print(results)


if __name__ == "__main__":
    main()

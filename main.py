from datasets import Dataset
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy, Faithfulness


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
    llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)
    )
    embedding = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    results = evaluate(
        dataset,
        metrics=[Faithfulness(), AnswerRelevancy()],
        llm=llm,
        embeddings=embedding,
    )

    print(results)


if __name__ == "__main__":
    main()

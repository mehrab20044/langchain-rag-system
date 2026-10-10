import json
from pathlib import Path
from statistics import mean

from sacrebleu import sentence_bleu

from langchain_rag_system.langchain_rag import (
    llm,
    output_parser,
    prompt_template,
    retrieve_result,
)

EVALUATION_DIR = Path(__file__).resolve().parent

generation_chain = prompt_template | llm | output_parser


def main():
    dataset = json.loads(
        (EVALUATION_DIR / "dataset.json").read_text(encoding="utf-8")
    )

    results = []

    for item in dataset:
        query = item["query"]
        retrieved = retrieve_result(query)

        answer = generation_chain.invoke(
            {
                "context": retrieved["text"],
                "query": query,
            }
        )

        relevant_ids = set(item["relevant_source_ids"])

        precision = None
        recall = None
        refusal_match = None

        if item["answerable"]:
            hit = int(retrieved["source_id"] in relevant_ids)
            precision = float(hit)
            recall = hit / len(relevant_ids)
        else:
            refusal_match = (
                answer.strip().casefold()
                == item["expected_answer"].strip().casefold()
            )

        results.append(
            {
                "query": query,
                "answerable": item["answerable"],
                "expected_answer": item["expected_answer"],
                "answer": answer,
                "retrieved_source_id": retrieved["source_id"],
                "context": retrieved["text"],
                "precision_at_1": precision,
                "recall_at_1": recall,
                "bleu": sentence_bleu(
                    answer,
                    [item["expected_answer"]],
                ).score,
                "refusal_exact_match": refusal_match,
            }
        )

        print(f"\nQuestion: {query}")
        print(f"Source: {retrieved['source_id']}")
        print(f"Answer: {answer}")

    answerable_results = [
        result for result in results if result["answerable"]
    ]

    summary = {
        "mean_precision_at_1": mean(
            result["precision_at_1"] for result in answerable_results
        ),
        "mean_recall_at_1": mean(
            result["recall_at_1"] for result in answerable_results
        ),
        "mean_bleu_answerable": mean(
            result["bleu"] for result in answerable_results
        ),
    }

    report = {
        "summary": summary,
        "results": results,
    }

    (EVALUATION_DIR / "results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nSummary:")
    print(json.dumps(summary, indent=2))
    print("\nSaved: evaluation/results.json")


if __name__ == "__main__":
    main()
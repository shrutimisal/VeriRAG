"""Human-readable explanations of each RAGAS metric (shown in the dashboard)."""

METRIC_INFO = {
    "faithfulness": {
        "title": "Faithfulness",
        "what": "Are the claims in the answer supported by the retrieved context?",
        "how": "An LLM splits the answer into statements and checks each against the retrieved chunks. "
               "Score = supported statements / total statements.",
        "low": "Low score = the answer contains hallucinated or unsupported content.",
    },
    "answer_relevancy": {
        "title": "Answer Relevancy",
        "what": "Does the answer actually address the question?",
        "how": "An LLM generates question(s) from the answer; the score is the embedding similarity "
               "between those and the original question.",
        "low": "Low score = answer is off-topic, incomplete or padded with irrelevant text.",
    },
    "context_precision": {
        "title": "Context Precision",
        "what": "Are the relevant chunks ranked near the top of the retrieved list?",
        "how": "An LLM judges each retrieved chunk as useful/not useful against the reference answer; "
               "precision is averaged with rank-awareness.",
        "low": "Low score = retrieval returns a lot of noise or ranks good chunks too low.",
    },
    "context_recall": {
        "title": "Context Recall",
        "what": "Did retrieval find all the information needed to produce the reference answer?",
        "how": "The reference answer is split into statements; each is checked for being attributable "
               "to the retrieved context.",
        "low": "Low score = important evidence was missed (increase top-k, improve chunking/embeddings).",
    },
}

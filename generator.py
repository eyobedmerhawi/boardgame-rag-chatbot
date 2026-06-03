from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL

_client = Groq(api_key=GROQ_API_KEY)


def generate_response(query, retrieved_chunks):
    """
    Generate a grounded answer from retrieved rule chunks.
    """
    if not retrieved_chunks:
        return (
            "I couldn't find anything relevant in the loaded rule books. "
            "Try rephrasing your question — or check that your ingestion pipeline is working."
        )

    context = "\n\n".join(
        f"[Game: {chunk['game']} | Distance: {chunk['distance']:.3f}]\n{chunk['text']}"
        for chunk in retrieved_chunks
    )

    system_prompt = """
You are RulesBot, a board game rules assistant.

Answer using only the rule text provided in the context.
Do not use outside knowledge.
Do not guess or invent rules.
If the answer is not contained in the provided context, say:
"I couldn't find that rule in the loaded rule books."

When you answer, clearly mention which game the rule comes from.
Keep the answer concise and helpful.
"""

    user_prompt = f"""
Retrieved rule text:
{context}

User question:
{query}
"""

    response = _client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content
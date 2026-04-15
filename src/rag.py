
from llm import llm


def ask(query,chunks):

    context = "\n\n".join(
        [f"Page {c['page']}:\n{c['text']}" for c in chunks]
    )

    prompt = f"""
You are an insurance assistant.

Answer ONLY from the given context.
If not found, say "Not mentioned in policy".

Context:
{context}

Question:
{query}

Answer:
"""

    response = llm.invoke(prompt)

    return response.content
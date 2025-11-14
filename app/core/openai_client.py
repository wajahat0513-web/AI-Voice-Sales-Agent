import os
from openai import OpenAI
from app.core.config import settings



client = OpenAI(api_key=settings.OPENAI_API_KEY)


def generate_response(prompt: str) -> str:
    """
    Generates a natural conversational AI response optimized for voice.
    """
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[
                {"role": "system", "content": "You are a natural-sounding AI voice sales assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.8,
            max_tokens=150,
        )

        return completion.choices[0].message.content.strip()

    except Exception as e:
        print(f"Error generating AI response: {e}")
        return "Sorry, there was a technical issue. Please hold on a moment."

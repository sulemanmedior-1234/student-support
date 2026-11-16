import streamlit as st
from groq import Groq

def generate_summary(prompt_context, fallback_text):
    """Generates an AI summary via Groq API if keys exist, else returns offline fallback."""
    try:
        api_key = st.secrets.get("GROQ_API_KEY")
        if not api_key:
            return fallback_text
            
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are an expert AI academic advisor and data science assistant summarizing student analytics clearly and professionally."},
                {"role": "user", "content": prompt_context}
            ],
            temperature=0.3,
            max_tokens=300
        )
        return completion.choices[0].message.content
    except Exception as e:
        # Graceful fallback if offline or API error occurs
        return f"{fallback_text}\n\n*(Note: Groq LLM summary unavailable offline/due to error: {str(e)})*"

from groq import Groq

client = Groq(
    api_key="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"
)

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "user",
            "content": "Hello! Tell me about melanoma in simple words."
        }
    ]
)

print(response.choices[0].message.content)
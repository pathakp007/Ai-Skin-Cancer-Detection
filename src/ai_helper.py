from groq import Groq

# ==========================================================
# Enter your Groq API Key here
# ==========================================================

client = Groq(
    api_key="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"
)

# ==========================================================
# AI Report Generator
# ==========================================================

def get_disease_information(disease, confidence):

    if confidence >= 80:
        confidence_text = (
            "The prediction confidence is high, but it is still not a confirmed diagnosis."
        )

    elif confidence >= 50:
        confidence_text = (
            "The prediction confidence is moderate. The result should be interpreted carefully."
        )

    else:
        confidence_text = (
            "The prediction confidence is low. The uploaded image may not closely match any of the trained disease categories."
        )

    prompt = f"""
You are an AI medical education assistant.

A TensorFlow image classification model predicted the following:

Disease:
{disease}

Confidence:
{confidence:.2f}%

Confidence Interpretation:
{confidence_text}

Generate a professional report in Markdown.

The report must contain the following headings exactly.

# 🩺 Disease Overview

Explain the disease in simple English.

# ⚠️ Common Symptoms

Give bullet points.

# 🔍 Possible Causes

Give bullet points.

# 💡 General Skin-care Advice

Provide only general skin-care suggestions.

Examples:

- Keep the skin clean
- Protect from excessive sunlight
- Avoid scratching the lesion

Never prescribe medicines.

Never recommend creams.

Never recommend homemade remedies.

Never claim that this is the correct disease.

# 🛡 Prevention Tips

Provide bullet points.

# 👨‍⚕️ When Should Someone Visit a Dermatologist?

Explain clearly.

# 📊 Confidence Interpretation

Explain what the confidence percentage means.

# ❗ Disclaimer

State clearly:

• This is an AI prediction.

• It is not a medical diagnosis.

• Only a qualified dermatologist can diagnose skin diseases.

Keep the answer between 250 and 400 words.

Use Markdown formatting.

Do not generate tables.
"""

    try:

        response = client.chat.completions.create(

            model="llama-3.3-70b-versatile",

            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional medical education assistant. "
                        "Provide educational information only. "
                        "Do not prescribe treatments or medicines."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.3,

            max_tokens=1000,

            top_p=0.9,

            stream=False
        )

        return response.choices[0].message.content

    except Exception as e:

        return f"""
# ❌ Error

Unable to generate AI report.

Reason:

{str(e)}

Please verify:

- Internet connection
- Groq API Key
- Groq API availability
"""
import os
import openai

# Load your API key from an environment variable or secret management service
openai.api_key = "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"
print(openai.Model.list())
response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {
      "role": "system",
      "content": "You will be provided with a tweet, and your task is to classify its sentiment as positive, neutral, or negative."
    },
    {
      "role": "user",
      "content": "I loved the new Batman movie!"
    }
  ],
  temperature=0,
  max_tokens=256
)

print(response)


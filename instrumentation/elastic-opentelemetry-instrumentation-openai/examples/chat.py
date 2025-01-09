import os

import openai

CHAT_MODEL = os.environ.get("TEST_CHAT_MODEL", "gpt-4o-mini")


def main():
    client = openai.Client()

    messages = [
        {
            "role": "user",
            "content": "Answer in up to 3 words: Which ocean contains Bouvet Island?",
        }
    ]

    chat_completion = client.chat.completions.create(model=CHAT_MODEL, messages=messages)
    print(chat_completion.choices[0].message.content)


if __name__ == "__main__":
    main()

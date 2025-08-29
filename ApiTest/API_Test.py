import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(
    base_url="https://models.github.ai/inference",
    api_key=os.getenv("GITHUB_API_KEY")
)
def chat_gpt(prompet):

        response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": ""},
            {"role": "user", "content": prompet}
        ],
        model="openai/gpt-4.1-mini",
        temperature=1,
        max_tokens=4096,
        top_p=1
        )
        return response.choices[0].message.content

# print(response.choices[0].message.content)
if __name__ == "__main__":
        while True :
                input_user = input('you: ')
                if  input_user.lower() in ['quit','bay','exist']:
                        break
                resp =chat_gpt(input_user)
                print('chatbot :',resp)
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
from src.utils import count_tokens_in_prompt

MAXIMUM_MO_TOKENS_PER_PROMPT = 10_000


client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
)

def get_openai_model_response(prompt: str, model_name: str = "gpt-4o"):
    no_of_token_in_prompt = count_tokens_in_prompt(prompt, model_name)
    if no_of_token_in_prompt > MAXIMUM_MO_TOKENS_PER_PROMPT:
        print(f"The prompt you are about to send is too large: {no_of_token_in_prompt}")
        raise ValueError
    try:
        # Send a request to the GPT-4 model
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},  # System message to set behavior
                {"role": "user", "content": prompt}  # User's prompt
            ],
            temperature=0,  # Adjust for creativity (0 = deterministic, 1 = very creative)
            max_tokens=1000  # Adjust for the length of the response
        )
        # Extract the content of the assistant's reply
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {e}"




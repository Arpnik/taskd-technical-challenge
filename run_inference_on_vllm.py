"""
Run inference using vLLM server API (for already running vllm serve).
This is more efficient than loading the model multiple times.

Usage:
    python vllm_client.py
    python vllm_client.py --host http://localhost:8000
    python vllm_client.py --max_tokens 1024 --temperature 0.5
"""

import requests
import json
import argparse

parser = argparse.ArgumentParser(description='Run inference via vLLM server')
parser.add_argument('--host', type=str, default="http://localhost:8000",
                    help='vLLM server host (default: http://localhost:8000)')
parser.add_argument('--max_tokens', type=int, default=512, help='max tokens (default 512)')
parser.add_argument('--temperature', type=float, default=0.01, help='temperature (default 0.01)')
args = parser.parse_args()

# ============================================================================
# Configure API endpoint
# ============================================================================
API_URL = f"{args.host}/v1/chat/completions"

print(f"Using vLLM server at: {args.host}")

# ============================================================================
# Helper Function: Call vLLM API
# ============================================================================
def query_model(question: str, max_tokens: int = 512, temperature: float = 0.01):
    """
    Send request to vLLM server using OpenAI-compatible API.
    """
    payload = {
        "model": "taskd_merged_model",  # Can be any string when using vllm serve
        "messages": [
            {"role": "user", "content": question}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.9,
    }

    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(API_URL, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        print(f"Error calling API: {e}")
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            print(f"Response: {e.response.text}")
        return None

# ============================================================================
# Interactive Mode - Continuous Prompting
# ============================================================================
print("\n" + "=" * 80)
print("INTERACTIVE MODE")
print("=" * 80)
print("Ask questions to the model. Type 'exit', 'quit', or 'q' to stop.")
print("=" * 80)

while True:
    try:
        user_input = input("\n> ").strip()

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        answer = query_model(user_input, args.max_tokens, args.temperature)

        if answer:
            print(f"\n{answer}")
        else:
            print("\n❌ Failed to get response")

    except KeyboardInterrupt:
        print("\n\nGoodbye!")
        break
    except EOFError:
        print("\n\nGoodbye!")
        break

print("\n" + "=" * 80)
print("SESSION ENDED")
print("=" * 80)
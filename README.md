# Technical Challenge: Deploy Llama 3.2:3B with vLLM

## Objective

Deploy the Llama 3.2:3B language model using vLLM and create an OpenAI-compatible API server that can respond to inference requests.

## Challenge Requirements

You must:

1. Install vLLM in the provided Docker environment
2. Deploy the `unsloth/Llama-3.2-3B-Instruct` model using vLLM's OpenAI-compatible server
3. Verify the deployment by querying the `/v1/models` endpoint
4. Test the model with a chat completion request

## Hints

### Research Topics

Before starting, read about:

- **Llama 3.2 3B**: Meta's 3-billion parameter instruction-tuned model
- **vLLM**: High-throughput and memory-efficient inference engine for LLMs
- **Unsloth**: Optimized model variants that don't require HuggingFace authentication

### Environment Notes

- The environment already has the correct PyTorch version installed
- You can install packages globally - this is a contained Docker environment
- Use the `unsloth/Llama-3.2-3B-Instruct` model to avoid HuggingFace login requirements
- You don't need to build vLLM from source - use pip installation
- Use host `0.0.0.0` since we're running in a Docker context
- Ignore the "Offline Batched Inference" section in vLLM docs
- Focus on the "OpenAI-Compatible Server" documentation

### How to Verify the Deployment

In a **new terminal**, verify the server is running and the model is loaded:

```bash
curl http://localhost:8000/v1/models
```

**Expected response format:**

```json
{
  "object": "list",
  "data": [
    {
      "id": "unsloth/Llama-3.2-3B-Instruct",
      "object": "model",
      "created": <timestamp>,
      "owned_by": "vllm",
      "root": "unsloth/Llama-3.2-3B-Instruct",
      "parent": null,
      "permission": [...]
    }
  ]
}
```

### How to Test Chat Completion

Install `jq` for JSON formatting (optional but recommended):

```bash
apt install jq
```

Send a test request to the chat completion endpoint:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "unsloth/Llama-3.2-3B-Instruct",
    "messages": [
      {"role": "user", "content": "Say hi"}
    ]
  }' | jq -r '.choices[0].message.content'
```

**Expected response:** A friendly greeting from the model.

### Common Issues

**Out of Memory Error:**

- Reduce `--gpu-memory-utilization` to 0.7 or 0.8
- Reduce `--max-model-len` to 2048

**Connection Refused:**

- Ensure you're using `0.0.0.0` as the host
- Check that port 8000 is not already in use
- Wait for the model to fully load before making requests

**Model Download Fails:**

- Verify internet connectivity
- Confirm you're using `unsloth/Llama-3.2-3B-Instruct` (no HuggingFace login needed)

## What's Next?

After completing this challenge, contact the challenge administrator to review your solution and proceed to the next stage.

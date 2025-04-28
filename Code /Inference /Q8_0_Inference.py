# from llama_cpp import Llama

# # Load the GGUF model (update path)
# model_path = "Llama-3.2-1B/Llama-3.2-1B-Q8_0.gguf"
# llm = Llama(model_path=model_path)

# # Generate a response
# response = llm("Hello, how are you?", max_tokens=100)
# print(response["choices"][0]["text"])

# import os
# os.environ["GGML_METAL_NO_LOG"] = "1"

from llama_cpp import Llama

llm = Llama(model_path="Llama-3.2-1B/Llama-3.2-1B-Q8_0.gguf", verbose=False, n_ctx=2048)

# Generate text
output = llm("i have a blister on my left hand. what is it ?",
             max_tokens= 100,
             temperature=0.1, # 0-1 0-more deterministic, 1-more creative
             repeat_penalty=1.1, # <1-free to repeat, >1 no repeat <2.5
            )
print(output["choices"][0]["text"]) # Extracts only the generated text

import torch
from transformers import pipeline

pipeline = pipeline(
    task="fill-mask",
    model="google-bert/bert-base-uncased",
    torch_dtype=torch.float16,
    device=0
)
#pipeline("Plants create [MASK] through a process known as photosynthesis.")


# Run inference
result = pipeline("Plants create [MASK] through a process known as photosynthesis.")

# Pretty print results
print("\nTop predictions:")
for i, pred in enumerate(result, 1):
    print(f"{i}. {pred['token_str']:<10} (score: {pred['score']:.4f})")

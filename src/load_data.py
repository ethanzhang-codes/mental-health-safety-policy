from datasets import load_dataset
import pandas as pd

# Dreaddit - stress detection from Reddit
dreaddit = load_dataset("asmaab/dreaddit_train")

# Emotion dataset - using confirmed working path
emotions = load_dataset("dair-ai/emotion", trust_remote_code=True)

# Mental health counseling conversations
counseling = load_dataset("Amod/mental_health_counseling_conversations")

print("All datasets loaded successfully")
print(f"\nDreaddit: {dreaddit}")
print(f"\nEmotions: {emotions}")
print(f"\nCounseling: {counseling}")

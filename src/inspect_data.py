from datasets import load_dataset
import pandas as pd

# Load all three
dreaddit = load_dataset("asmaab/dreaddit_train")
emotions = load_dataset("dair-ai/emotion")
counseling = load_dataset("Amod/mental_health_counseling_conversations")

# --- Inspect Dreaddit ---
print("=== DREADDIT ===")
dreaddit_df = pd.DataFrame(dreaddit['train'])
print(dreaddit_df['label'].value_counts())
print(dreaddit_df['post'].iloc[0][:200])  # preview first post

# --- Inspect Emotions ---
print("\n=== EMOTIONS ===")
emotions_df = pd.DataFrame(emotions['train'])
print(emotions_df['label'].value_counts())
print(emotions_df['text'].iloc[0])

# --- Inspect Counseling ---
print("\n=== COUNSELING ===")
counseling_df = pd.DataFrame(counseling['train'])
print(counseling_df.columns.tolist())
print(counseling_df['Context'].iloc[0][:200])

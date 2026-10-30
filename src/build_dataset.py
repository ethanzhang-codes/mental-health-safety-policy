from datasets import load_dataset
import pandas as pd

# Load all three
dreaddit = load_dataset("asmaab/dreaddit_train")
emotions = load_dataset("dair-ai/emotion")
counseling = load_dataset("Amod/mental_health_counseling_conversations")

# --- DREADDIT ---
# yes = stressed → medium risk (1)
# no  = not stressed → low risk (0)
dreaddit_df = pd.DataFrame(dreaddit['train'])
dreaddit_df = dreaddit_df[['post', 'label']].rename(columns={'post': 'text'})
dreaddit_df['risk_severity'] = dreaddit_df['label'].map({'yes': 1, 'no': 0})
dreaddit_df = dreaddit_df[['text', 'risk_severity']]
print(f"Dreaddit: {len(dreaddit_df)} rows")

# --- EMOTIONS ---
# Emotion label mapping for dair-ai/emotion:
# 0=sadness, 1=joy, 2=love, 3=anger, 4=fear, 5=surprise
# joy + love → low risk (0)
# anger + surprise → low risk (0)  
# sadness + fear → medium risk (1)
emotions_df = pd.DataFrame(emotions['train'])
emotion_to_risk = {
    0: 1,  # sadness → medium
    1: 0,  # joy → low
    2: 0,  # love → low
    3: 0,  # anger → low (frustration, not crisis)
    4: 1,  # fear → medium
    5: 0,  # surprise → low
}
emotions_df['risk_severity'] = emotions_df['label'].map(emotion_to_risk)
emotions_df = emotions_df[['text', 'risk_severity']]
print(f"Emotions: {len(emotions_df)} rows")

# --- COUNSELING ---
# All counseling conversations → high risk (2)
# Use the Context column (user message)
counseling_df = pd.DataFrame(counseling['train'])
counseling_df = counseling_df[['Context']].rename(columns={'Context': 'text'})
counseling_df['risk_severity'] = 2
print(f"Counseling: {len(counseling_df)} rows")

# --- COMBINE ---
combined = pd.concat([dreaddit_df, emotions_df, counseling_df], ignore_index=True)
combined = combined.dropna(subset=['text', 'risk_severity'])
combined = combined[combined['text'].str.strip() != '']
combined['risk_severity'] = combined['risk_severity'].astype(int)

# --- REPORT ---
print(f"\nTotal examples: {len(combined)}")
print(f"\nLabel distribution:")
print(combined['risk_severity'].value_counts().sort_index())
print("\nLabel meaning: 0=low risk, 1=medium risk, 2=high risk")

# --- SAVE ---
combined.to_csv("combined_dataset.csv", index=False)
print("\nSaved to combined_dataset.csv")
print("\nSample from each class:")
for level in [0, 1, 2]:
    sample = combined[combined['risk_severity'] == level]['text'].iloc[0][:150]
    print(f"\nLevel {level}: {sample}")

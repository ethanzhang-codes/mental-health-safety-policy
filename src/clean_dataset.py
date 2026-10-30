import pandas as pd

df = pd.read_csv("combined_dataset.csv")

# Remove "Post: " prefix left over from Dreaddit
df['text'] = df['text'].str.replace(r'^Post:\s*', '', regex=True)

# Remove very short texts (under 10 characters, likely junk)
df = df[df['text'].str.len() >= 10]

df.to_csv("combined_dataset.csv", index=False)

print(f"Cleaned dataset: {len(df)} rows")
print(f"\nLabel distribution:")
print(df['risk_severity'].value_counts().sort_index())
print(f"\nSample level 0: {df[df['risk_severity']==0]['text'].iloc[0][:150]}")
print(f"\nSample level 1: {df[df['risk_severity']==1]['text'].iloc[0][:150]}")
print(f"\nSample level 2: {df[df['risk_severity']==2]['text'].iloc[0][:150]}")

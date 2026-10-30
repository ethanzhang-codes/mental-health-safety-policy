import os

import pandas as pd
from datasets import load_dataset


DATASET_NAME = "to-be/annomi-motivational-interviewing-therapy-conversations"
OUTPUT_PATH = "./data/annomi_turns.csv"


def main():
    dataset = load_dataset(DATASET_NAME, split="train")

    rows = []
    first_conversation = None

    for conversation in dataset:
        conversation_id = conversation["id"]
        turns = conversation["conversations"]

        if first_conversation is None:
            first_conversation = conversation

        for turn_id, turn in enumerate(turns):
            rows.append({
                "conversation_id": conversation_id,
                "turn_id": turn_id,
                "speaker_role": turn["from"],
                "text": turn["value"],
            })

    output_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)

    num_conversations = len(dataset)
    num_turns = len(output_df)
    avg_turns = num_turns / num_conversations if num_conversations else 0

    print(f"Number of conversations: {num_conversations}")
    print(f"Number of total turns: {num_turns}")
    print(f"Average turns per conversation: {avg_turns:.2f}")
    print(f"Saved processed turns to: {OUTPUT_PATH}")

    print("\nFirst full conversation preview:")
    print(f"conversation_id: {first_conversation['id']}")
    for turn_id, turn in enumerate(first_conversation["conversations"]):
        print(f"[{turn_id}] {turn['from']}: {turn['value']}")


if __name__ == "__main__":
    main()

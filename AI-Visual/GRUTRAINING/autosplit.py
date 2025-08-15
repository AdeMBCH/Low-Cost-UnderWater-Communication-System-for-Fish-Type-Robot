import pandas as pd
import os
import sys

# === CONFIG ===
INPUT_CSV = sys.argv[1] if len(sys.argv) > 1 else "training_data_5.csv"
OUTPUT_DIR = "split_messages"
LABEL_COLUMN = "Label"

# Extraire l'ID du fichier source (ex: 1 depuis training_data_1.csv)
base_name = os.path.basename(INPUT_CSV)
file_id = os.path.splitext(base_name)[0].split("_")[-1]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Charger les données
df = pd.read_csv(INPUT_CSV)
df = df.dropna(subset=[LABEL_COLUMN]).reset_index(drop=True)

# Grouper par label consécutif
segments = []
current_label = None
current_chunk = []

for _, row in df.iterrows():
    label = row[LABEL_COLUMN]
    if label != current_label:
        if current_chunk:
            segments.append((current_label, pd.DataFrame(current_chunk)))
        current_label = label
        current_chunk = [row]
    else:
        current_chunk.append(row)

# Ajouter le dernier segment
if current_chunk:
    segments.append((current_label, pd.DataFrame(current_chunk)))

# Sauvegarder chaque segment
total = 0
counters = {}
for label, segment in segments:
    label_str = str(label).replace(".", "_")  # éviter les problèmes de nom
    counters[label_str] = counters.get(label_str, 0) + 1
    outname = f"training_data_{label_str}_{file_id}_{counters[label_str]:03d}.csv"
    segment.to_csv(os.path.join(OUTPUT_DIR, outname), index=False)
    total += 1

print(f"✅ {total} séquences extraites depuis '{INPUT_CSV}' → dossier '{OUTPUT_DIR}'")
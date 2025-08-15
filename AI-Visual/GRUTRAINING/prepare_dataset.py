import torch
from torch.utils.data import TensorDataset
import random
import numpy as np

# === CONFIGURATION ===
NUM_SEQUENCES = 80000            # Beaucoup de messages simulés
BITS_PER_MESSAGE = 16            # Plus long = plus robuste
NOISE_LEVEL = 0.15               # 15% de frames altérées (bruit/drop)
TARGET_SEQ_LEN = BITS_PER_MESSAGE // 2
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# === Encodage des bits → LED
BITS_TO_LED = {
    (0, 0): [0, 0],
    (0, 1): [0, 1],
    (1, 0): [1, 0],
    (1, 1): [1, 1],
}

def add_noise(sequence, noise_level):
    """Ajoute du bruit en flippant ou supprimant aléatoirement des frames."""
    new_seq = []
    for led in sequence:
        if random.random() < noise_level:
            noise_type = random.choice(['flip', 'drop'])
            if noise_type == 'flip':
                noisy = [bit ^ random.randint(0, 1) for bit in led]
                new_seq.append(noisy)
            elif noise_type == 'drop':
                continue  # saute cette frame
        else:
            new_seq.append(led)
    return new_seq

def generate_sequence(bits_per_message):
    bits = [random.randint(0, 1) for _ in range(bits_per_message)]
    led_seq = []
    for i in range(0, len(bits), 2):
        pair = (bits[i], bits[i+1])
        led_seq.append(BITS_TO_LED[pair])
    return bits, led_seq

# === GENERATE DATASET
X_list = []
y_list = []

for _ in range(NUM_SEQUENCES):
    bits, clean_seq = generate_sequence(BITS_PER_MESSAGE)
    noisy_seq = add_noise(clean_seq, NOISE_LEVEL)

    # Pad or truncate pour forcer la bonne longueur
    while len(noisy_seq) < TARGET_SEQ_LEN:
        noisy_seq.append([0, 0])
    noisy_seq = noisy_seq[:TARGET_SEQ_LEN]

    X_list.append(noisy_seq)
    y_list.append(bits)

X_tensor = torch.tensor(X_list, dtype=torch.float32)  # (N, seq_len, 2)
y_tensor = torch.tensor(y_list, dtype=torch.long)     # (N, bits)

dataset = TensorDataset(X_tensor, y_tensor)
torch.save(dataset, "bit_recovery_dataset_Gpu_ASCII.pt")

print(f"✅ Dataset saved: {len(dataset)} sequences")
print(f"   ➤ Input shape : {X_tensor.shape}")
print(f"   ➤ Target shape: {y_tensor.shape}")

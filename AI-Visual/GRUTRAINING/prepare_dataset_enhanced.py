import torch
from torch.utils.data import TensorDataset
import random
import numpy as np

# === CONFIGURATION ===
NUM_SEQUENCES = 80000
BITS_PER_MESSAGE = 16
NOISE_LEVEL = 0.15
TARGET_SEQ_LEN = BITS_PER_MESSAGE // 2
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

BITS_TO_LED = {
    (0, 0): [0, 0],
    (0, 1): [0, 1],
    (1, 0): [1, 0],
    (1, 1): [1, 1],
}

def add_noise(sequence, noise_level):
    new_seq = []
    for led in sequence:
        if random.random() < noise_level:
            noise_type = random.choice(['flip', 'drop'])
            if noise_type == 'flip':
                noisy = [bit ^ random.randint(0, 1) for bit in led]
                new_seq.append(noisy)
            elif noise_type == 'drop':
                continue
        else:
            new_seq.append(led)
    return new_seq

def text_to_bits(text):
    return [int(b) for c in text for b in format(ord(c), '08b')]

def generate_sequence():
    number = random.randint(10, 99)
    str_number = str(number)  # e.g. "73"
    bits = text_to_bits(str_number)  # 16 bits
    led_seq = [BITS_TO_LED[(bits[i], bits[i+1])] for i in range(0, len(bits), 2)]
    return bits, led_seq

# === GENERATE DATASET
X_list = []
y_list = []

for _ in range(NUM_SEQUENCES):
    bits, clean_seq = generate_sequence()
    noisy_seq = add_noise(clean_seq, NOISE_LEVEL)

    # Pad/truncate to ensure correct length
    while len(noisy_seq) < TARGET_SEQ_LEN:
        noisy_seq.append([0, 0])
    noisy_seq = noisy_seq[:TARGET_SEQ_LEN]

    X_list.append(noisy_seq)
    y_list.append(bits)

X_tensor = torch.tensor(X_list, dtype=torch.float32)
y_tensor = torch.tensor(y_list, dtype=torch.long)

dataset = TensorDataset(X_tensor, y_tensor)
torch.save(dataset, "bit_recovery_dataset_enhanced2.pt")

print(f"✅ Dataset saved: {len(dataset)} sequences")
print(f"   ➤ Input shape : {X_tensor.shape}")
print(f"   ➤ Target shape: {y_tensor.shape}")

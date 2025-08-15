import torch
from torch.utils.data import TensorDataset
import random
import numpy as np
import string

# === CONFIGURATION ===
NUM_SEQUENCES = 10000
MAX_MESSAGE_CHARS = 200
NOISE_LEVEL = 0.15
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

def generate_sequence(max_chars=MAX_MESSAGE_CHARS):
    length = random.randint(1, max_chars)
    message = ''.join(random.choices(string.printable, k=length))  # Full ASCII
    bits = text_to_bits(message)
    led_seq = [BITS_TO_LED[(bits[i], bits[i+1])] for i in range(0, len(bits), 2)]
    return bits, led_seq

# === GENERATE DATASET
X_list = []
y_list = []

MAX_BITS = MAX_MESSAGE_CHARS * 8      # 1600 bits max
MAX_LED_SEQ_LEN = MAX_BITS // 2       # 800 LED steps max

for _ in range(NUM_SEQUENCES):
    bits, clean_seq = generate_sequence()
    noisy_seq = add_noise(clean_seq, NOISE_LEVEL)

    # Pad/truncate inputs and targets
    while len(noisy_seq) < MAX_LED_SEQ_LEN:
        noisy_seq.append([0, 0])
    noisy_seq = noisy_seq[:MAX_LED_SEQ_LEN]

    while len(bits) < MAX_BITS:
        bits.append(0)
    bits = bits[:MAX_BITS]

    X_list.append(noisy_seq)
    y_list.append(bits)

X_tensor = torch.tensor(X_list, dtype=torch.float32)     # [N, 800, 2]
y_tensor = torch.tensor(y_list, dtype=torch.long)        # [N, 1600]

dataset = TensorDataset(X_tensor, y_tensor)
torch.save(dataset, "bit_recovery_dataset_more_full_ascii_ultimate.pt")

print(f"✅ Dataset saved: {len(dataset)} sequences")
print(f"   ➤ Input shape : {X_tensor.shape}")  # [80000, 800, 2]
print(f"   ➤ Target shape: {y_tensor.shape}")  # [80000, 1600]

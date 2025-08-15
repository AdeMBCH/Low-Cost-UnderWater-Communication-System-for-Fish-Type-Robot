import torch
import torch.nn as nn
import random
import numpy as np

# === CONFIGURATION ===
MODEL_PATH = "gru_bit_model_more_paper_ultimate.pt"
HIDDEN_SIZE = 256
NUM_LAYERS = 2
INPUT_SIZE = 2
OUTPUT_BITS = 1600
NOISE_LEVEL = 0.15  # Match training

# === LED Encoding ===
BITS_TO_LED = {
    (0, 0): [0, 0],
    (0, 1): [0, 1],
    (1, 0): [1, 0],
    (1, 1): [1, 1],
}
LED_TO_BITS = {tuple(v): list(k) for k, v in BITS_TO_LED.items()}

# === Utility Functions ===
def text_to_bits(text):
    return [int(b) for char in text.encode('utf-8') for b in format(char, '08b')]

def bits_to_text(bits):
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        if len(byte) < 8:
            continue
        chars.append(chr(int("".join(str(b) for b in byte), 2)))
    return "".join(chars)

def bits_to_leds(bits):
    leds = []
    for i in range(0, len(bits), 2):
        pair = (bits[i], bits[i+1]) if i+1 < len(bits) else (bits[i], 0)
        leds.append(BITS_TO_LED[pair])
    return leds

def add_noise(seq, noise_level):
    noisy = []
    for led in seq:
        if random.random() < noise_level:
            noise_type = random.choice(["flip", "drop"])
            if noise_type == "flip":
                noisy.append([bit ^ random.randint(0, 1) for bit in led])
            elif noise_type == "drop":
                continue
        else:
            noisy.append(led)
    # Pad back to original length
    while len(noisy) < len(seq):
        noisy.append([0, 0])
    return noisy[:len(seq)]

# === Model Definition ===
class BitGRU(nn.Module):
    def __init__(self):
        super(BitGRU, self).__init__()
        self.gru = nn.GRU(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, batch_first=True)
        self.fc = nn.Linear(HIDDEN_SIZE, OUTPUT_BITS)

    def forward(self, x):
        _, h_n = self.gru(x)
        return self.fc(h_n[-1])

# === Load Model ===
model = BitGRU()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

# === User Input ===
message = input("💬 Entrez un message (jusqu'à ~200 caractères) : ").strip() + '\x00'
bits = text_to_bits(message)
led_seq = bits_to_leds(bits)

# === Pad to match 800 LED steps (max)
while len(led_seq) < 800:
    led_seq.append([0, 0])
led_seq = led_seq[:800]

# === Noise Injection ===
noisy_led_seq = add_noise(led_seq, NOISE_LEVEL)

# === Inference ===
x_tensor = torch.tensor([noisy_led_seq], dtype=torch.float32)
with torch.no_grad():
    output = model(x_tensor)
    pred_bits = (torch.sigmoid(output) > 0.5).int().squeeze().tolist()
    pred_bits = pred_bits[:len(bits)]

# === Output Results ===
print(f"\n🔢 Bits initiaux ({len(bits)}): {bits}")
print(f"📼 LED bruitées : {noisy_led_seq[:10]} ...")
print(f"🎯 Bits prédits ({len(pred_bits)}): {pred_bits[:len(bits)]} ...")
print(f"🗣️  Message récupéré : {bits_to_text(pred_bits)}")

import torch
import torch.nn as nn
import random
import numpy as np

# === CONFIG ===
MODEL_PATH = "gru_numnum_gpu_test.pt"
BITS_PER_CHAR = 8
NOISE_LEVEL = 0.1  # 10% frames bruitées
HIDDEN_SIZE = 256
INPUT_SIZE = 2
NUM_LAYERS = 2

# === Encodage LED (QPSK simplifié) ===
BITS_TO_LED = {
    (0, 0): [0, 0],
    (0, 1): [0, 1],
    (1, 0): [1, 0],
    (1, 1): [1, 1],
}
LED_TO_BITS = {tuple(v): list(k) for k, v in BITS_TO_LED.items()}

# === Fonctions auxiliaires ===
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
    # Padding if needed
    target_len = len(seq)
    while len(noisy) < target_len:
        noisy.append([0, 0])
    return noisy[:target_len]

# === GRU Model ===
class BitGRU(nn.Module):
    def __init__(self):
        super(BitGRU, self).__init__()
        self.gru = nn.GRU(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, batch_first=True)
        self.fc = nn.Linear(HIDDEN_SIZE, 16)  # 16 bits (2 char)

    def forward(self, x):
        _, h_n = self.gru(x)
        return self.fc(h_n[-1])

# === CHARGE LE MODÈLE ===
model = BitGRU()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# === ENTREZ UN MESSAGE ===
message = input("💬 Entrez un message court (ex. 'OK') : ").strip()
bits = text_to_bits(message)
led_seq = bits_to_leds(bits)
noisy_led_seq = add_noise(led_seq, NOISE_LEVEL)

# === PRÉPARATION DU TENSOR ===
x_tensor = torch.tensor([noisy_led_seq], dtype=torch.float32)  # shape: (1, seq_len, 2)
# === PRÉDICTION ===
with torch.no_grad():
    output = model(x_tensor)
    pred_bits = (torch.sigmoid(output) > 0.5).int().squeeze().tolist()
print(pred_bits)
# === AFFICHAGE ===
print(f"\n🔢 Bits initiaux : {bits}")
print(f"📼 LED bruitées : {noisy_led_seq}")
print(f"🎯 Bits prédits : {pred_bits}")
print(f"🗣️  Message récupéré : {bits_to_text(pred_bits)}")

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from typing import List, Tuple
import pandas as pd
import seaborn as sns
import os

# ========== CONFIGURABLE PARAMETERS ==========
MODEL_PATH = "gru_numnum_gpu_test.pt"
MESSAGES = ["01", "23", "49", "78", "65"]
#MESSAGES = ["23", "14", "06", "42", "15"]
NOISE_LEVELS = np.round(np.linspace(0.0, 1.0, 21), 2).tolist()
FLICKER_RANGE = (0, 3)  # 0 to 3 frames occluded
NUM_TRIALS = 1000  # per condition
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PLOTS = True
PLOT_PREFIX = "results/"
os.makedirs(PLOT_PREFIX, exist_ok=True)
# =============================================

BITS_TO_LED = {(0, 0): [0, 0], (0, 1): [0, 1], (1, 0): [1, 0], (1, 1): [1, 1]}
LED_TO_BITS = {tuple(v): list(k) for k, v in BITS_TO_LED.items()}
SEQ_BITS = 16
SEQ_LEN = SEQ_BITS // 2
INPUT_SIZE = 2
HIDDEN_SIZE = 256
NUM_LAYERS = 1


def text_to_bits(text: str) -> List[int]:
    return [int(b) for c in text.encode() for b in format(c, '08b')]

def bits_to_text(bits: List[int]) -> str:
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        if len(byte) == 8:
            chars.append(chr(int(''.join(map(str, byte)), 2)))
    return ''.join(chars)

def bits_to_leds(bits: List[int]) -> List[List[int]]:
    return [BITS_TO_LED.get((bits[i], bits[i+1]), [0, 0]) for i in range(0, len(bits), 2)]

def leds_to_bits(led_seq: List[List[int]]) -> List[int]:
    return [bit for led in led_seq for bit in LED_TO_BITS.get(tuple(led), [0, 0])]

def digit_similarity(gt: str, pred: str) -> float:
    if len(gt) != len(pred):
        return 0.0
    return sum(g == p for g, p in zip(gt, pred)) / len(gt)

def add_noise(seq: List[List[int]], noise_level=0.1, flicker_frames=0) -> List[List[int]]:
    total_frames = len(seq)
    num_corrupted = int(noise_level * total_frames)
    indices = list(range(total_frames))
    corrupted_indices = set(random.sample(indices, num_corrupted))
    noisy = []
    for i, led in enumerate(seq):
        if i in corrupted_indices:
            if random.random() < 0.5:
                noisy.append([bit ^ random.randint(0, 1) for bit in led])
            else:
                continue
        else:
            noisy.append(led)
    for _ in range(flicker_frames):
        if noisy:
            idx = random.randint(0, len(noisy) - 1)
            noisy[idx] = [0, 0]
    while len(noisy) < SEQ_LEN:
        noisy.append([0, 0])
    return noisy[:SEQ_LEN]

class BitGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, batch_first=True)
        self.fc = nn.Linear(HIDDEN_SIZE, SEQ_BITS)

    def forward(self, x):
        _, h_n = self.gru(x)
        return self.fc(h_n[-1])

def run_inference(model, led_seq: List[List[int]]) -> Tuple[List[int], float]:
    x_tensor = torch.tensor([led_seq], dtype=torch.float32).to(DEVICE)
    start = time.time()
    with torch.no_grad():
        output = model(x_tensor)
        pred_bits = (torch.sigmoid(output) > 0.5).int().squeeze().tolist()
    latency = time.time() - start
    return pred_bits, latency

def baseline_decode(led_seq: List[List[int]]) -> List[int]:
    return leds_to_bits(led_seq)

def evaluate_condition(model, message, noise_level, flicker_frames):
    gt_bits = text_to_bits(message)
    led_seq = bits_to_leds(gt_bits)
    noisy_seq = add_noise(led_seq, noise_level=noise_level, flicker_frames=flicker_frames)
    pred_bits, latency = run_inference(model, noisy_seq)
    baseline_bits = baseline_decode(noisy_seq)
    decoded = bits_to_text(pred_bits)
    baseline_decoded = bits_to_text(baseline_bits)
    return {
        "bit_acc": sum(p == g for p, g in zip(pred_bits, gt_bits)) / len(gt_bits),
        "msg_acc": int(decoded == message),
        "msg_similarity": digit_similarity(message, decoded),
        "baseline_bit_acc": sum(p == g for p, g in zip(baseline_bits, gt_bits)) / len(gt_bits),
        "baseline_msg_acc": int(baseline_decoded == message),
        "baseline_msg_similarity": digit_similarity(message, baseline_decoded),
        "latency": latency
    }

def run_experiments():
    model = BitGRU().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    results = []
    for noise in NOISE_LEVELS:
        print(f"\n🔁 Testing noise level: {noise:.2f}")
        for _ in range(NUM_TRIALS):
            for message in MESSAGES:
                res = evaluate_condition(model, message, noise, random.randint(*FLICKER_RANGE))
                res.update({"noise": noise, "message": message})
                results.append(res)
    return results

def plot_and_summarize(results: List[dict]):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    df = pd.DataFrame(results)

    # Select only numeric columns for aggregation
    numeric_cols = df.select_dtypes(include=['number']).columns
    summary = df.groupby("noise")[numeric_cols].agg(['mean', 'std']).round(4)
    summary.to_csv(PLOT_PREFIX + "summary_table.csv")

    metrics = [
        ("bit_acc", "Bit Accuracy"),
        ("msg_acc", "Message Accuracy"),
        ("msg_similarity", "Message Similarity"),
        ("latency", "Inference Latency (s)")
    ]

    sns.set(style="whitegrid", context="talk")
    for key, label in metrics:
        plt.figure(figsize=(8, 5))
        means = df.groupby("noise")[key].mean()
        stds = df.groupby("noise")[key].std()
        plt.errorbar(means.index, means, yerr=stds, fmt='-o', capsize=3, label="GRU")

        if "baseline_" + key in df.columns:
            baseline_means = df.groupby("noise")["baseline_" + key].mean()
            baseline_stds = df.groupby("noise")["baseline_" + key].std()
            plt.errorbar(baseline_means.index, baseline_means, yerr=baseline_stds, fmt='-x', capsize=3, label="Baseline")

        plt.title(f"{label} vs Noise Level")
        plt.xlabel("Noise Level")
        plt.ylabel(label)
        plt.legend()
        plt.tight_layout()
        if SAVE_PLOTS:
            plt.savefig(f"{PLOT_PREFIX}{key}_vs_noise.png")
        plt.close()

    print("\n📊 Summary table saved to:", PLOT_PREFIX + "summary_table.csv")
    return df


if __name__ == "__main__":
    results = run_experiments()
    df = plot_and_summarize(results)
    df.to_csv(PLOT_PREFIX + "full_results.csv", index=False)
    print("\n✅ All experiments and plots complete.")

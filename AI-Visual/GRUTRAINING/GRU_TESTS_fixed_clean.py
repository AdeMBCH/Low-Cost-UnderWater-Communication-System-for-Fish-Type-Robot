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
# MESSAGES = ["23", "14", "06", "42", "15"]
NOISE_LEVELS = np.round(np.linspace(0.0, 1.0, 21), 2).tolist()
FLICKER_RANGE = (0, 3)  # 0 to 3 frames occluded
NUM_TRIALS = 100  # per condition
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PLOTS = True
PLOT_PREFIX = "results/"
os.makedirs(PLOT_PREFIX, exist_ok=True)
SEED = 42  # for reproducibility
# =============================================

# ====== REPRODUCIBILITY ======
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ====== MAPPINGS / CONSTANTS ======
BITS_TO_LED = {(0, 0): [0, 0], (0, 1): [0, 1], (1, 0): [1, 0], (1, 1): [1, 1]}
LED_TO_BITS = {tuple(v): list(k) for k, v in BITS_TO_LED.items()}
SEQ_BITS = 16
SEQ_LEN = SEQ_BITS // 2
INPUT_SIZE = 2
HIDDEN_SIZE = 256
NUM_LAYERS = 1

# ====== UTILITIES ======
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

# Helpers for symbol sequences
from collections import deque

def leds_to_syms(led_seq: List[List[int]]) -> List[Tuple[int, int]]:
    return [tuple(x) for x in led_seq]

# ====== NOISE MODEL (symbol-level) ======
def add_noise(seq: List[List[int]], noise_level=0.1, flicker_frames=0) -> List[List[int]]:
    """
    Protocol preserved: one frame = one 2-bit symbol.
    - With probability proportional to noise_level, a frame is corrupted: either flipped (per-bit xor with {0,1}) or dropped.
    - 'flicker_frames' additional frames are forced to [0,0] to mimic occlusion.
    Deleted frames are later padded with [0,0] to keep fixed SEQ_LEN.
    """
    total_frames = len(seq)
    num_corrupted = int(noise_level * total_frames)
    indices = list(range(total_frames))
    corrupted_indices = set(random.sample(indices, num_corrupted))

    noisy = []
    for i, led in enumerate(seq):
        if i in corrupted_indices:
            if random.random() < 0.5:
                # flip (per bit) with p=0.5
                noisy.append([bit ^ random.randint(0, 1) for bit in led])
            else:
                # drop the frame
                continue
        else:
            noisy.append(led)

    # additional flicker to [0,0]
    for _ in range(flicker_frames):
        if noisy:
            idx = random.randint(0, len(noisy) - 1)
            noisy[idx] = [0, 0]

    # pad / truncate to fixed SEQ_LEN
    while len(noisy) < SEQ_LEN:
        noisy.append([0, 0])
    return noisy[:SEQ_LEN]

# ====== GRU MODEL ======
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

# ====== DIRECT (NON-SEQUENTIAL) BASELINE ======
def baseline_decode(led_seq: List[List[int]]) -> List[int]:
    return leds_to_bits(led_seq)

# ====== HMM (VITERBI) BASELINE ======
STATES = [(0, 0), (0, 1), (1, 0), (1, 1)]
STATE_IDX = {s: i for i, s in enumerate(STATES)}

def hamming2(a: Tuple[int,int], b: Tuple[int,int]) -> int:
    return int(a[0] != b[0]) + int(a[1] != b[1])

def build_hmm_params(noise_level: float, flicker_frames: int, seq_len: int,
                     stay_prob: float = 0.5, eps: float = 1e-9):
    """
    One symbol per frame; 4-state HMM. Transitions near-uniform (stay_prob=0.5)
    to avoid over-smoothing when your symbol stream changes often.
    Emissions from independent bit flips; (0,0) gets extra mass from p_drop.
    """
    p_flip = 0.5 * noise_level
    p_drop = min(0.99, 0.5 * noise_level + (flicker_frames / max(1, seq_len)))

    pi = np.full(len(STATES), 1.0 / len(STATES))
    A = np.full((len(STATES), len(STATES)), (1.0 - stay_prob) / (len(STATES) - 1))
    np.fill_diagonal(A, stay_prob)

    E = {}
    for s in STATES:
        scores = []
        for o in STATES:
            d = hamming2(s, o)
            base = ((1 - p_flip) ** (2 - d)) * (p_flip ** d)
            if o == (0, 0):
                base += p_drop
            scores.append(max(eps, base))
        scores = np.array(scores)
        scores /= scores.sum()
        E[s] = scores
    return pi, A, E

def hmm_decode(led_seq: List[List[int]], noise_level: float, flicker_frames: int) -> List[int]:
    T = len(led_seq)
    obs = [tuple(x) for x in led_seq]
    pi, A, E = build_hmm_params(noise_level, flicker_frames, T)

    logA = np.log(A + 1e-12)
    logpi = np.log(pi + 1e-12)

    logB = np.full((T, len(STATES)), -np.inf)
    for t in range(T):
        o = obs[t]
        o_idx = STATE_IDX.get(o, STATE_IDX[(0, 0)])
        for s_idx, s in enumerate(STATES):
            logB[t, s_idx] = np.log(E[s][o_idx] + 1e-12)

    delta = np.full((T, len(STATES)), -np.inf)
    psi = np.full((T, len(STATES)), -1, dtype=int)
    delta[0, :] = logpi + logB[0, :]

    for t in range(1, T):
        for j in range(len(STATES)):
            prev = delta[t - 1, :] + logA[:, j]
            psi[t, j] = int(np.argmax(prev))
            delta[t, j] = prev[psi[t, j]] + logB[t, j]

    path = np.zeros(T, dtype=int)
    path[-1] = int(np.argmax(delta[-1, :]))
    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]

    sym_bits = []
    for idx in path:
        sym_bits.extend(list(STATES[idx]))
    return sym_bits[:SEQ_BITS]

# ====== HMM (FORWARD–BACKWARD / BCJR) BASELINE ======
# Compute posteriors p(state_t | observations) and take per-time argmax (MAP symbols).

def fb_decode(led_seq: List[List[int]], noise_level: float, flicker_frames: int) -> List[int]:
    T = len(led_seq)
    obs = [tuple(x) for x in led_seq]
    pi, A, E = build_hmm_params(noise_level, flicker_frames, T)

    # Emission matrix B[t, s] = P(o_t | s)
    B = np.zeros((T, len(STATES)))
    for t in range(T):
        o = obs[t]
        o_idx = STATE_IDX.get(o, STATE_IDX[(0, 0)])
        for s_idx, s in enumerate(STATES):
            B[t, s_idx] = E[s][o_idx]

    # Forward
    alpha = np.zeros((T, len(STATES)))
    alpha[0, :] = pi * B[0, :]
    alpha[0, :] /= max(1e-12, alpha[0, :].sum())
    for t in range(1, T):
        alpha[t, :] = (alpha[t-1, :].dot(A)) * B[t, :]
        alpha[t, :] /= max(1e-12, alpha[t, :].sum())

    # Backward
    beta = np.zeros((T, len(STATES)))
    beta[-1, :] = 1.0
    for t in range(T-2, -1, -1):
        beta[t, :] = (A * B[t+1, :]).dot(beta[t+1, :])
        beta[t, :] /= max(1e-12, beta[t, :].sum())

    gamma = alpha * beta
    gamma /= np.maximum(1e-12, gamma.sum(axis=1, keepdims=True))

    path = np.argmax(gamma, axis=1)
    sym_bits = []
    for idx in path:
        sym_bits.extend(list(STATES[idx]))
    return sym_bits[:SEQ_BITS]

# ====== TEMPORAL MEDIAN (MAJORITY) FILTER BASELINE ======
# Classic non-parametric smoothing per bit with sliding window size k (odd).

def median_filter_decode(led_seq: List[List[int]], k: int = 3) -> List[int]:
    assert k % 2 == 1, "k must be odd"
    T = len(led_seq)
    half = k // 2
    arr = np.array(led_seq)  # shape (T, 2)
    smoothed = np.zeros_like(arr)
    for t in range(T):
        i0, i1 = max(0, t - half), min(T, t + half + 1)
        window = arr[i0:i1]
        # majority per bit
        smoothed[t, 0] = 1 if window[:, 0].sum() * 2 >= len(window) else 0
        smoothed[t, 1] = 1 if window[:, 1].sum() * 2 >= len(window) else 0
    return leds_to_bits(smoothed.tolist())

# ====== EVALUATION ======
def evaluate_condition(model, message, noise_level, flicker_frames):
    gt_bits = text_to_bits(message)
    led_seq = bits_to_leds(gt_bits)
    noisy_seq = add_noise(led_seq, noise_level=noise_level, flicker_frames=flicker_frames)

    # GRU
    pred_bits, latency = run_inference(model, noisy_seq)

    # Direct baseline
    base_bits = baseline_decode(noisy_seq)

    # HMM baselines
    hmm_bits = hmm_decode(noisy_seq, noise_level=noise_level, flicker_frames=flicker_frames)
    fb_bits  = fb_decode(noisy_seq,  noise_level=noise_level, flicker_frames=flicker_frames)

    # Median filter baseline (k=3)
    med_bits = median_filter_decode(noisy_seq, k=3)

    decoded = bits_to_text(pred_bits)
    base_dec = bits_to_text(base_bits)
    hmm_dec  = bits_to_text(hmm_bits)
    fb_dec   = bits_to_text(fb_bits)
    med_dec  = bits_to_text(med_bits)

    return {
        "bit_acc": np.mean([p == g for p, g in zip(pred_bits, gt_bits)]),
        "msg_acc": int(decoded == message),
        "msg_similarity": digit_similarity(message, decoded),

        "baseline_bit_acc": np.mean([p == g for p, g in zip(base_bits, gt_bits)]),
        "baseline_msg_acc": int(base_dec == message),
        "baseline_msg_similarity": digit_similarity(message, base_dec),

        "hmm_bit_acc": np.mean([p == g for p, g in zip(hmm_bits, gt_bits)]),
        "hmm_msg_acc": int(hmm_dec == message),
        "hmm_msg_similarity": digit_similarity(message, hmm_dec),

        "fb_bit_acc": np.mean([p == g for p, g in zip(fb_bits, gt_bits)]),
        "fb_msg_acc": int(fb_dec == message),
        "fb_msg_similarity": digit_similarity(message, fb_dec),

        "med_bit_acc": np.mean([p == g for p, g in zip(med_bits, gt_bits)]),
        "med_msg_acc": int(med_dec == message),
        "med_msg_similarity": digit_similarity(message, med_dec),

        "latency": latency,
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


# ====== PLOTTING / SUMMARY ======
def plot_and_summarize(results: List[dict]):
    df = pd.DataFrame(results)

    # numeric-only aggregation
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

        for prefix, marker, name in [
            ("baseline_", 'x', "Direct"),
            ("hmm_",      's', "HMM (Viterbi)"),
            ("fb_",       'd', "HMM (FB/BCJR)"),
            ("med_",      '^', "Median k=3"),
        ]:
            col = prefix + key
            if col in df.columns:
                m = df.groupby("noise")[col].mean()
                s = df.groupby("noise")[col].std()
                plt.errorbar(m.index, m, yerr=s, fmt=f'-{marker}', capsize=3, label=name)

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


# ====== MAIN ======
if __name__ == "__main__":
    results = run_experiments()
    df = plot_and_summarize(results)
    df.to_csv(PLOT_PREFIX + "full_results.csv", index=False)
    print("\n✅ All experiments and plots complete.")

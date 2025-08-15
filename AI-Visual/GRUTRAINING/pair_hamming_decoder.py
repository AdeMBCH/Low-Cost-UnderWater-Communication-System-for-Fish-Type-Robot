import random

# === CONFIG ===
BITS_PER_CHAR = 8
NOISE_LEVEL = 0.1   # 10% des frames bruitées
MAX_SHIFT = 2       # tolérance au décalage temporel pour le mode codebook
USE_CODEBOOK = True # si True, on fait aussi un décodage par codebook (messages candidats)

# Messages candidats (facultatif / pour le mode codebook)
CANDIDATES = ["01", "23", "49", "78", "65", "OK", "LE", "RT", "VX", "15"]

# === Encodage LED (QPSK simplifié) ===
BITS_TO_LED = {
    (0, 0): [0, 0],
    (0, 1): [0, 1],
    (1, 0): [1, 0],
    (1, 1): [1, 1],
}
LED_TO_BITS = {tuple(v): list(k) for k, v in BITS_TO_LED.items()}
CANONICAL_LEDS = [ [0,0], [0,1], [1,0], [1,1] ]  # ordre fixe pour recherche NN

# === Fonctions auxiliaires ===
def text_to_bits(text):
    return [int(b) for ch in text.encode('utf-8') for b in format(ch, '08b')]

def bits_to_text(bits):
    # Tronque à l'octet le plus proche vers le bas
    n = len(bits) - (len(bits) % 8)
    bits = bits[:n]
    out = []
    for i in range(0, n, 8):
        byte = bits[i:i+8]
        out.append(chr(int("".join(str(b) for b in byte), 2)))
    return "".join(out)

def bits_to_leds(bits):
    leds = []
    for i in range(0, len(bits), 2):
        if i+1 < len(bits):
            pair = (bits[i], bits[i+1])
        else:
            pair = (bits[i], 0)
        leds.append(BITS_TO_LED[pair])
    return leds

def add_noise(seq, noise_level):
    noisy = []
    for led in seq:
        if random.random() < noise_level:
            if random.random() < 0.5:
                # flip aléatoire de 1 ou 2 bits
                noisy.append([b ^ random.randint(0,1) for b in led])
            else:
                # drop: on supprime la frame
                continue
        else:
            noisy.append(led)
    # Padding pour garder la même longueur
    target_len = len(seq)
    while len(noisy) < target_len:
        noisy.append([0,0])
    return noisy[:target_len]

# === Hamming pair helpers ===
def pair_hamming(a, b):
    # a,b: listes [x,y] de 0/1
    return int(a[0] != b[0]) + int(a[1] != b[1])

def nearest_led_pair(obs):
    # renvoie la paire canonique la plus proche (distance de Hamming)
    best = CANONICAL_LEDS[0]
    best_d = pair_hamming(obs, best)
    for ref in CANONICAL_LEDS[1:]:
        d = pair_hamming(obs, ref)
        if d < best_d:
            best, best_d = ref, d
    return best

def shift_seq(seq, k):
    pad = [0,0]
    if k > 0:
        return [pad]*k + seq[:-k] if k < len(seq) else [pad]*len(seq)
    elif k < 0:
        k = -k
        return seq[k:] + [pad]*k if k < len(seq) else [pad]*len(seq)
    return seq[:]

def best_distance_over_shifts(obs, ref, max_shift=2):
    best = None
    for s in range(-max_shift, max_shift+1):
        shifted = shift_seq(ref, s)
        d = sum(pair_hamming(o, r) for o, r in zip(obs, shifted))
        best = d if best is None else min(best, d)
    return best if best is not None else 0

# === Décodages ===
def decode_framewise_pair_hamming(noisy_led_seq):
    # Décodage indépendant frame par frame (NN Hamming vers {00,01,10,11})
    recovered_leds = [nearest_led_pair(led) for led in noisy_led_seq]
    bits = []
    for led in recovered_leds:
        bits.extend(LED_TO_BITS[tuple(led)])
    return bits

def decode_codebook(noisy_led_seq, candidates, max_shift=2):
    # Pour chaque message candidat : on reconstruit sa séquence LED idéale
    # et on choisit celui qui minimise la distance de Hamming paire-à-paire
    best = None
    best_bits = None
    best_msg = None
    for msg in candidates:
        bits = text_to_bits(msg)
        ref_leds = bits_to_leds(bits)
        # ajuster longueur à celle observée
        if len(ref_leds) < len(noisy_led_seq):
            ref_leds = ref_leds + [[0,0]] * (len(noisy_led_seq) - len(ref_leds))
        elif len(ref_leds) > len(noisy_led_seq):
            ref_leds = ref_leds[:len(noisy_led_seq)]
        d = best_distance_over_shifts(noisy_led_seq, ref_leds, max_shift=max_shift)
        if (best is None) or (d < best):
            best = d
            best_bits = bits[:len(noisy_led_seq)*2]
            best_msg = msg
    return best_bits if best_bits is not None else [0]*(len(noisy_led_seq)*2), best_msg

# === I/O ===
if __name__ == "__main__":
    print("Pair-Hamming Decoder (sans GRU)\n")
    message = input("💬 Entrez un message court (ex. 'OK'): ").strip()
    if not message:
        message = "OK"

    bits = text_to_bits(message)
    led_seq = bits_to_leds(bits)
    noisy_led_seq = add_noise(led_seq, NOISE_LEVEL)

    # Décodage 1: framewise (pair Hamming)
    pred_bits_framewise = decode_framewise_pair_hamming(noisy_led_seq)
    pred_text_framewise = bits_to_text(pred_bits_framewise)

    # Décodage 2 (optionnel): codebook + tolérance de décalage
    if USE_CODEBOOK:
        pred_bits_codebook, best_msg = decode_codebook(noisy_led_seq, CANDIDATES, max_shift=MAX_SHIFT)
        pred_text_codebook = bits_to_text(pred_bits_codebook)
    else:
        pred_bits_codebook, pred_text_codebook, best_msg = None, None, None

    # === AFFICHAGE ===
    print(f"\n🔢 Bits initiaux : {bits}")
    print(f"📼 LED bruitées : {noisy_led_seq}")
    print(f"\n🎯 Framewise (pair Hamming) -> bits : {pred_bits_framewise}")
    print(f"🗣️  Framewise -> texte : {pred_text_framewise}")
    if USE_CODEBOOK:
        print(f"\n📚 Codebook (tolérance ±{MAX_SHIFT} frames) -> meilleur candidat : {best_msg}")
        print(f"🎯 Codebook -> bits : {pred_bits_codebook}")
        print(f"🗣️  Codebook -> texte : {pred_text_codebook}")

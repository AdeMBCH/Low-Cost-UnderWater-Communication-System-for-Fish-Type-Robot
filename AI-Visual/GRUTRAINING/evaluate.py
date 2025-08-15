import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import numpy as np

# === CONFIG ===
MODEL_PATH = "gru_bit_model.pt"
DATASET_PATH = "bit_recovery_dataset.pt"
BATCH_SIZE = 64
HIDDEN_SIZE = 256  # ⚠️ Doit correspondre à ton entraînement
NUM_LAYERS = 2     # ⚠️ Idem
INPUT_SIZE = 2
OUTPUT_BITS = 16

# === MODEL DEFINITION ===
class BitGRU(nn.Module):
    def __init__(self):
        super(BitGRU, self).__init__()
        self.gru = nn.GRU(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, batch_first=True)
        self.fc = nn.Linear(HIDDEN_SIZE, OUTPUT_BITS)

    def forward(self, x):
        _, h_n = self.gru(x)
        return self.fc(h_n[-1])

# === LOAD MODEL ===
model = BitGRU()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# === LOAD DATASET ===
dataset = torch.load(DATASET_PATH)
test_loader = DataLoader(dataset, batch_size=BATCH_SIZE)

# === EVALUATION ===
y_true = []
y_pred = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_y = batch_y.int()
        outputs = model(batch_x)
        preds = (torch.sigmoid(outputs) > 0.5).int()
        y_true.extend(batch_y.numpy())
        y_pred.extend(preds.numpy())

y_true = np.array(y_true).flatten()
y_pred = np.array(y_pred).flatten()

# === BIT-LEVEL ACCURACY ===
bit_accuracy = np.mean(y_true == y_pred)
print(f"🧠 Bit-level Accuracy: {bit_accuracy*100:.2f}%")

# === CONFUSION MATRIX ===
cm = confusion_matrix(y_true, y_pred)
print("\n📊 Confusion Matrix:")
print(cm)

# === CLASSIFICATION REPORT ===
print("\n📈 Classification Report:")
print(classification_report(y_true, y_pred, target_names=["0", "1"]))

# === DISPLAY MATRIX ===
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["0", "1"])
disp.plot()
plt.title("Bit-level Confusion Matrix")
plt.grid(False)
plt.show()

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device utilisé : {device}")
torch.set_num_threads(8)
torch.set_num_interop_threads(2)

# === CONFIG ===
DATASET_FILE = "bit_recovery_dataset_Gpu_ASCII.pt"
MODEL_SAVE_PATH = "gru_ASCII_gpu_test.pt"
BATCH_SIZE = 32
EPOCHS = 100 #300
LEARNING_RATE = 0.001
HIDDEN_SIZE = 256
NUM_LAYERS = 2
INPUT_SIZE = 2   # LED1, LED2
OUTPUT_BITS = 16  # nombre total de bits à prédire

# === Load Dataset ===
dataset = torch.load(DATASET_FILE, weights_only=False)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# === GRU Model ===
class BitGRU(nn.Module):
    def __init__(self):
        super(BitGRU, self).__init__()
        self.gru = nn.GRU(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, batch_first=True)
        self.fc = nn.Linear(HIDDEN_SIZE, OUTPUT_BITS)

    def forward(self, x):
        _, h_n = self.gru(x)  # h_n: (num_layers, batch, hidden)
        out = self.fc(h_n[-1])  # Use last hidden state
        return out  # shape: (batch, OUTPUT_BITS)

model = BitGRU().to(device)
criterion = nn.BCEWithLogitsLoss()  # bits → [0,1]
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# === Training ===
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device).float()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

# === Evaluation ===
model.eval()
y_true = []
y_pred = []
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        outputs = model(batch_x)
        preds = (torch.sigmoid(outputs) > 0.5).long()
        y_true.extend(batch_y.cpu().numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())

# === Bit-level Accuracy ===
total_bits = len(y_true) * OUTPUT_BITS
correct_bits = sum(
    int(pred_bit == true_bit)
    for pred_seq, true_seq in zip(y_pred, y_true)
    for pred_bit, true_bit in zip(pred_seq, true_seq)
)
bit_acc = correct_bits / total_bits
print(f"🧠 Bit Accuracy: {bit_acc*100:.2f}%")

# === Confusion Matrix + Metrics ===
y_true_flat = np.array(y_true).flatten()
y_pred_flat = np.array(y_pred).flatten()

print("\n📊 Confusion Matrix:")
cm = confusion_matrix(y_true_flat, y_pred_flat)
print(cm)

print("\n📈 Classification Report:")
print(classification_report(y_true_flat, y_pred_flat, target_names=["0", "1"]))

# === Display Confusion Matrix ===
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["0", "1"])
disp.plot()
plt.title("Bit-level Confusion Matrix")
plt.grid(False)
plt.show()


# === Save Model ===
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"✅ Modèle sauvegardé : {MODEL_SAVE_PATH}")

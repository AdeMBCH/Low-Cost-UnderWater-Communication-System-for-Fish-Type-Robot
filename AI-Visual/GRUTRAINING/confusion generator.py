import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# === Replace this with your confusion matrix ===
conf_matrix = np.array([[106431, 21916],
                        [21875, 105778]])

# === Plot the confusion matrix ===
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

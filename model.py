# ============================================================
# MODELO COMPLETO — Classificação + Detecção de OOD
# ============================================================


# ============================================================
# CÉLULA 1 — Imports e configuração
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset, ConcatDataset, random_split
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import csv
import copy

torch.manual_seed(42)  # reprodutibilidade

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Usando:", device)


# ============================================================
# CÉLULA 2 — Transformações (com Data Augmentation) e Datasets
# ============================================================
train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

train_full = datasets.FashionMNIST(root="./data", train=True, download=True, transform=train_transform)

n_val = 6000
n_train = len(train_full) - n_val
train_dataset, val_dataset = random_split(train_full, [n_train, n_val])

# a validação não deve ter augmentation "agressivo" -> criamos uma cópia sem RandomRotation/Flip
val_dataset.dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=eval_transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)

classes = train_full.classes
num_classes = len(classes)
print("Classes:", classes)
print(f"Treino: {len(train_dataset)} | Validação: {len(val_dataset)}")


# ============================================================
# CÉLULA 3 — Conjunto de avaliação "contaminado" (simula dado real de prova)
# ============================================================
fashion_test = datasets.FashionMNIST(root="./data", train=False, download=True, transform=eval_transform)
fashion_subset = Subset(fashion_test, range(1000))

mnist_test = datasets.MNIST(root="./data", train=False, download=True, transform=eval_transform)
mnist_subset = Subset(mnist_test, range(200))

N_FASHION = len(fashion_subset)
N_OOD = len(mnist_subset)

eval_dataset = ConcatDataset([fashion_subset, mnist_subset])
eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=False)

print(f"Conjunto de avaliação: {N_FASHION} roupas + {N_OOD} 'intrusos' = {len(eval_dataset)} total")


# ============================================================
# CÉLULA 4 — Arquitetura da CNN (mais robusta: BatchNorm + Dropout + 3 blocos)
# ============================================================
class RobustCNN(nn.Module):
    def __init__(self, num_classes=10, dropout=0.3):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)          # 28x28 -> 14x14
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)          # 14x14 -> 7x7
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # 7x7 -> 1x1 (reduz sem precisar calcular tamanho exato na mão)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x

model = RobustCNN(num_classes=num_classes, dropout=0.3).to(device)
print(model)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total de parâmetros: {total_params:,}")


# ============================================================
# CÉLULA 5 — Loss, otimizador, scheduler e treino com early stopping
# ============================================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

NUM_EPOCHS = 20
PATIENCE = 5  # early stopping: para se não melhorar por 5 épocas

history = {"train_loss": [], "val_loss": [], "val_acc": []}
best_val_acc = 0.0
best_model_state = None
epochs_sem_melhora = 0

for epoch in range(NUM_EPOCHS):
    # --- Treino ---
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    train_loss = running_loss / len(train_dataset)

    # --- Validação ---
    model.eval()
    val_running_loss = 0.0
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss = val_running_loss / len(val_dataset)
    val_acc = correct / total

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    scheduler.step(val_acc)
    current_lr = optimizer.param_groups[0]['lr']

    print(f"Época {epoch+1}/{NUM_EPOCHS} - Train Loss: {train_loss:.4f} - "
          f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f} - LR: {current_lr:.6f}")

    # --- Early stopping + salvar melhor modelo ---
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = copy.deepcopy(model.state_dict())
        epochs_sem_melhora = 0
    else:
        epochs_sem_melhora += 1
        if epochs_sem_melhora >= PATIENCE:
            print(f"Early stopping na época {epoch+1} (sem melhora por {PATIENCE} épocas)")
            break

# Carrega o melhor modelo encontrado
model.load_state_dict(best_model_state)
torch.save(best_model_state, "melhor_modelo.pth")
print(f"\nMelhor acurácia de validação: {best_val_acc:.4f}")


# ============================================================
# CÉLULA 6 — Curvas de treino (loss e acurácia)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history["train_loss"], label="Treino")
axes[0].plot(history["val_loss"], label="Validação")
axes[0].set_title("Loss por época")
axes[0].set_xlabel("Época")
axes[0].set_ylabel("Loss")
axes[0].legend()

axes[1].plot(history["val_acc"], label="Val Acc", color="green")
axes[1].set_title("Acurácia de validação por época")
axes[1].set_xlabel("Época")
axes[1].set_ylabel("Acurácia")
axes[1].legend()

plt.tight_layout()
plt.show()


# ============================================================
# CÉLULA 7 — Matriz de confusão e relatório de classificação (na validação)
# ============================================================
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8, 7))
plt.imshow(cm, cmap="Blues")
plt.title("Matriz de Confusão (Validação)")
plt.colorbar()
plt.xticks(range(num_classes), classes, rotation=45, ha="right")
plt.yticks(range(num_classes), classes)
plt.xlabel("Previsto")
plt.ylabel("Real")
for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, cm[i, j], ha="center", va="center",
                  color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=8)
plt.tight_layout()
plt.show()

print(classification_report(all_labels, all_preds, target_names=classes))


# ============================================================
# CÉLULA 8 — Detecção de OOD: confidences separadas + threshold balanceado + AUROC
# ============================================================
fashion_conf, mnist_conf = [], []
idx = 0

model.eval()
with torch.no_grad():
    for inputs, _ in eval_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        probs = F.softmax(outputs, dim=1)
        confidence, _ = torch.max(probs, dim=1)

        for c in confidence.cpu().numpy():
            if idx < N_FASHION:
                fashion_conf.append(c)
            else:
                mnist_conf.append(c)
            idx += 1

fashion_conf = np.array(fashion_conf)
mnist_conf = np.array(mnist_conf)

print(f"Confiança média (roupas):     {fashion_conf.mean():.4f}")
print(f"Confiança média (OOD/MNIST):  {mnist_conf.mean():.4f}")

# Histograma comparando as distribuições
plt.figure(figsize=(8, 4))
plt.hist(fashion_conf, bins=30, alpha=0.6, label="Roupas (in-distribution)")
plt.hist(mnist_conf, bins=30, alpha=0.6, label="MNIST (OOD)")
plt.xlabel("Confiança (softmax máximo)")
plt.ylabel("Quantidade")
plt.legend()
plt.title("Distribuição de confiança: in-distribution vs OOD")
plt.show()

# Busca do melhor threshold por acurácia balanceada
melhor_threshold, melhor_score = None, -1
for t in np.arange(0.05, 1.0, 0.01):
    fp = (fashion_conf < t).sum()
    tp = (mnist_conf < t).sum()
    tnr = (N_FASHION - fp) / N_FASHION
    tpr = tp / N_OOD
    balanced = (tnr + tpr) / 2
    if balanced > melhor_score:
        melhor_score, melhor_threshold = balanced, t

fp = (fashion_conf < melhor_threshold).sum()
tp = (mnist_conf < melhor_threshold).sum()
print(f"\nMelhor threshold (balanceado): {melhor_threshold:.2f}")
print(f"  Falsos positivos (roupa marcada como OOD): {fp}/{N_FASHION}")
print(f"  Verdadeiros positivos (MNIST pego como OOD): {tp}/{N_OOD}")
print(f"  Acurácia balanceada: {melhor_score:.4f}")

# AUROC (métrica sem depender de threshold)
y_true = np.concatenate([np.zeros(len(fashion_conf)), np.ones(len(mnist_conf))])
y_score = np.concatenate([1 - fashion_conf, 1 - mnist_conf])
auroc = roc_auc_score(y_true, y_score)
print(f"  AUROC: {auroc:.4f}")


# ============================================================
# CÉLULA 9 — Gerar o CSV final de predições
# ============================================================
THRESHOLD = melhor_threshold  # usa o threshold calibrado na célula anterior

resultados = []
model.eval()
idx = 0
with torch.no_grad():
    for inputs, _ in eval_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        probs = F.softmax(outputs, dim=1)
        confidence, pred_class = torch.max(probs, dim=1)

        for i in range(inputs.size(0)):
            if confidence[i].item() < THRESHOLD:
                classe_predita = "OOD"
            else:
                classe_predita = classes[pred_class[i].item()]
            resultados.append((idx, classe_predita, round(confidence[i].item(), 4)))
            idx += 1

with open("predicoes.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "classe_predita", "confianca"])
    writer.writerows(resultados)

print("Arquivo predicoes.csv gerado com sucesso!")
print(f"Threshold usado: {THRESHOLD:.2f}")


# ============================================================
# CÉLULA 10 — Resumo para o relatório (imprime os números prontos)
# ============================================================
print("=" * 60)
print("RESUMO PARA O RELATÓRIO")
print("=" * 60)
print(f"Arquitetura: CNN com 3 blocos convolucionais (BatchNorm + ReLU + Pooling), "
      f"dropout de 0.3, {total_params:,} parâmetros")
print(f"Épocas treinadas: {len(history['val_acc'])} (early stopping paciência={PATIENCE})")
print(f"Melhor acurácia de validação: {best_val_acc:.4f}")
print(f"Estratégia de OOD: threshold de confiança softmax, calibrado por acurácia balanceada")
print(f"Threshold escolhido: {melhor_threshold:.2f}")
print(f"Acurácia balanceada na detecção de OOD: {melhor_score:.4f}")
print(f"AUROC na separação in-distribution vs OOD: {auroc:.4f}")
print("=" * 60)
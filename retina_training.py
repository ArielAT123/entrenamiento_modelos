# =========================================================
# 0. DEPENDENCIAS Y GOOGLE DRIVE
# =========================================================
import os
import json
from roboflow import Roboflow
import torch
import torchvision
from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.transforms import v2
from torch.utils.data import DataLoader

# Intentar montar Google Drive (requiere interacción manual del usuario si se ejecuta en Colab)
try:
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE_SAVE_DIR = '/content/drive/MyDrive/Modelos_Cacao_Aug_Correct'
except Exception as e:
    print(f"\nAdvertencia: No se pudo montar Google Drive ({e}).")
    print("Se guardarán los modelos localmente. Asegðrate de descargarlos al final.")
    DRIVE_SAVE_DIR = './Modelos_Cacao_Aug_Correct'

os.makedirs(DRIVE_SAVE_DIR, exist_ok=True)

# Inicializar Roboflow y descargar el dataset en formato COCO (NUEVO DATASET CON AUGMENTATION CORRECTO)
rf = Roboflow(api_key="Ik6eEYG0FuBkzGPLZSDK")
project = rf.workspace("ariels-workspace-b3hov").project("dataset_cocoa_disease_aug_ahorasi")
version = project.version(1)
dataset = version.download("coco")

# Rutas extraídas automáticamente
DATASET_DIR = dataset.location
TRAIN_IMG_DIR = os.path.join(DATASET_DIR, "train")
TRAIN_ANN_FILE = os.path.join(DATASET_DIR, "train", "_annotations.coco.json")

# =========================================================
# 1. INSPECCIóN DE LA ESTRUCTURA DEL DATASET (JSON)
# =========================================================
print("\n--- Inspeccionando el archivo de anotaciones COCO ---")
if os.path.exists(TRAIN_ANN_FILE):
    with open(TRAIN_ANN_FILE, 'r') as f:
        coco_data = json.load(f)
    print("Claves principales del JSON:", coco_data.keys())
    if 'annotations' in coco_data and len(coco_data['annotations']) > 0:
        sample_ann = coco_data['annotations'][0]
        print("Ejemplo de Bounding Box (xmin, ymin, width, height):", sample_ann.get('bbox'))
        print("Categoría ID:", sample_ann.get('category_id'))
    else:
        print("No se encontraron anotaciones en el JSON.")
else:
    print("Archivo de anotaciones no encontrado.")
print("-----------------------------------------------------\n")

# =========================================================
# 2. HIPERPARáMETROS ESTáNDAR Y PARA GPU A100
# =========================================================
EPOCHS = 150
BATCH_SIZE = 16         # Subido a 16 para aprovechar los 40GB/80GB de la A100
IMAGE_SIZE = (640, 640)
LEARNING_RATE = 1e-4    # Aðn más reducido (1e-5) para extrema estabilidad
WEIGHT_DECAY = 1e-4     # Prevención de sobreajuste

# Determinar el nðmero dinámico de clases desde el JSON para evitar errores OutOfBounds
if os.path.exists(TRAIN_ANN_FILE):
    with open(TRAIN_ANN_FILE, 'r') as f:
        temp_data = json.load(f)
    NUM_CLASSES = max(cat['id'] for cat in temp_data['categories']) + 1
else:
    NUM_CLASSES = 6
print(f"Nðmero de clases ajustado dinámicamente a: {NUM_CLASSES} (Fondo incluido)")

# =========================================================
# 3. DATASET PERSONALIZADO PARA RETINANET Y COCO
# =========================================================
class CocoaRetinaNetDataset(torchvision.datasets.CocoDetection):
    def __init__(self, root, annFile, transforms=None):
        super().__init__(root, annFile)
        self.custom_transforms = transforms

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)

        boxes = []
        labels = []

        # Convertir COCO [xmin, ymin, width, height] a PyTorch [xmin, ymin, xmax, ymax]
        for obj in target:
            xmin, ymin, w, h = obj['bbox']
            # FILTRO DE SEGURIDAD: Evitar cajas con área <= 0 que causan NaNs
            if w > 0 and h > 0:
                boxes.append([xmin, ymin, xmin + w, ymin + h])
                labels.append(obj['category_id'])

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        tv_boxes = torchvision.tv_tensors.BoundingBoxes(
            boxes, format="XYXY", canvas_size=(img.height, img.width)
        )

        my_target = {
            "boxes": tv_boxes,
            "labels": labels,
            "image_id": torch.tensor([self.ids[idx]])
        }

        if self.custom_transforms is not None:
            img, my_target = self.custom_transforms(img, my_target)

        return img, my_target

# Transformaciones estrictas (Sin aumentos extra para no dañar la señal de la enfermedad)
strict_transforms = v2.Compose([
    v2.ToImage(),
    v2.Resize(size=IMAGE_SIZE, antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.SanitizeBoundingBoxes(),
])

def collate_fn(batch):
    return tuple(zip(*batch))

# =========================================================
# 4. CARGA DE DATOS Y MODELO (Optimizado para A100)
# =========================================================
train_dataset = CocoaRetinaNetDataset(
    root=TRAIN_IMG_DIR,
    annFile=TRAIN_ANN_FILE,
    transforms=strict_transforms
)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
use_amp = device.type == 'cuda'

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=2,       # Ajustado a 2 para evitar advertencias de sobrecarga
    pin_memory=use_amp   # Acelera la transferencia de datos RAM -> GPU si hay CUDA
)

model = retinanet_resnet50_fpn_v2(weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT)

# Reemplazar completamente la cabecera de clasificación para el nuevo nðmero de clases
in_features = 256 # Por defecto en RetinaNet FPN
num_anchors = model.head.classification_head.num_anchors
model.head.classification_head = RetinaNetClassificationHead(
    in_channels=in_features,
    num_anchors=num_anchors,
    num_classes=NUM_CLASSES
)
model.to(device)

# =========================================================
# 5. OPTIMIZADOR Y BUCLE CON PRECISIóN MIXTA Y DRIVE
# =========================================================
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Inicializar Scaler de manera condicional (sólo si hay CUDA)
scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

print(f"Iniciando entrenamiento acelerado en {device} por {EPOCHS} épocas...")

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0

    for images, targets in train_loader:
        # Cargar a GPU de forma no bloqueante si aplica
        images = list(image.to(device, non_blocking=use_amp) for image in images)
        targets = [{k: v.to(device, non_blocking=use_amp) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        # Contexto AMP para máxima velocidad condicional
        with torch.amp.autocast(device.type, enabled=use_amp):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # Retropropagación escalada
        scaler.scale(losses).backward()

        # Desescalar gradientes para poder hacer clipping si usamos AMP
        if use_amp:
            scaler.unscale_(optimizer)

        # Recorte de gradientes para evitar explosión (NaNs)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        epoch_loss += losses.item()

    print(f"Época {epoch+1}/{EPOCHS} | Pérdida total: {epoch_loss/len(train_loader):.4f}")

    # Guardar el modelo en Google Drive (o local) cada 10 épocas
    if (epoch + 1) % 10 == 0:
        save_path = os.path.join(DRIVE_SAVE_DIR, f"retinanet_cacao_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"[*] Modelo guardado en: {save_path}")

print("¡Entrenamiento finalizado!")

# Guardar el modelo final
final_save_path = os.path.join(DRIVE_SAVE_DIR, "retinanet_cacao_final.pth")
torch.save(model.state_dict(), final_save_path)
print(f"Modelo final guardado en: {final_save_path}")
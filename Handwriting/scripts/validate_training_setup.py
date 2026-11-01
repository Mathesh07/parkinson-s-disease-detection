import sys
from pathlib import Path
hw_dir = Path(__file__).resolve().parent.parent
if str(hw_dir) not in sys.path:
    sys.path.insert(0, str(hw_dir))

from dataset import build_manifest, create_patient_level_split, create_dataloaders
from model import ViTBinaryClassifier

manifest, summary = build_manifest()
manifest = create_patient_level_split(manifest)
train_loader, val_loader, test_loader = create_dataloaders(manifest, batch_size=4, image_size=224, num_workers=0)
print('train batches', len(train_loader), 'val batches', len(val_loader), 'test batches', len(test_loader))
model = ViTBinaryClassifier().eval()
print('model loaded, classifier weight shape', tuple(model.classifier.weight.shape))
images, labels = next(iter(train_loader))
print('batch image shape', images.shape, 'batch labels', labels)

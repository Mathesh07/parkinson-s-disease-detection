import sys
sys.path.append(r'd:\Materials\SEM7\Project Work 1\Implementation\Handwriting')
from dataset import build_manifest, create_patient_level_split
import pandas as pd

manifest, summary = build_manifest()
manifest = create_patient_level_split(manifest)
patient_level = manifest.groupby('ID_PATIENT', as_index=False).first()
print('patient-level split counts:')
print(patient_level['split'].value_counts().to_string())
print('\npatient-level split by class:')
print(pd.crosstab(patient_level['split'], patient_level['target']).to_string())
print('\nimage-level split counts:')
print(manifest['split'].value_counts().to_string())

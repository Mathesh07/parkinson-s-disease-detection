import os, pandas as pd, glob
from collections import Counter
root = r'd:\Materials\SEM7\Project Work 1\Implementation\Handwriting\NewHandPD'
for csv_name in ['NewSpiral.csv','NewMeander.csv']:
    path = os.path.join(root,csv_name)
    print(f'=== {csv_name} ===')
    df = pd.read_csv(path)
    print('columns:', list(df.columns))
    print('shape:', df.shape)
    print(df.head(10).to_string(index=False))
    print()
    if 'CLASS_TYPE' in df.columns:
        print('CLASS_TYPE counts:')
        print(df['CLASS_TYPE'].value_counts(dropna=False).to_string())
        print()
    if 'ID_PATIENT' in df.columns:
        print('ID_PATIENT sample:', df['ID_PATIENT'].head(10).tolist())
        print('unique patients:', df['ID_PATIENT'].nunique())
        print('images per patient distribution:')
        counts = df['ID_PATIENT'].value_counts()
        print(counts.describe().to_string())
        print(counts.head(10).to_string())
        print()
    if 'IMAGE_NAME' in df.columns:
        print('IMAGE_NAME sample:', df['IMAGE_NAME'].head(10).tolist())
    print()

# verify image existence based on image_name column and likely folder names
for csv_name in ['NewSpiral.csv','NewMeander.csv']:
    path = os.path.join(root,csv_name)
    df = pd.read_csv(path)
    print(f'=== verifying image files for {csv_name} ===')
    image_names = df['IMAGE_NAME'].astype(str).tolist()
    found = 0
    missing = []
    for name in image_names:
        candidates = []
        # direct in root
        candidates.append(os.path.join(root, name))
        # inside subfolders based on common naming
        for folder in ['HealthySpiral','HealthyMeander','HealthyCircle','PatientSpiral','PatientMeander','PatientCircle']:
            candidates.append(os.path.join(root, folder, name))
        # also maybe same for HealthySignal/PatientSignal
        for folder in ['HealthySignal','PatientSignal']:
            candidates.append(os.path.join(root, folder, name))
        if any(os.path.exists(c) for c in candidates):
            found += 1
        else:
            missing.append(name)
    print('found:', found, 'missing:', len(missing))
    if missing:
        print('missing examples:', missing[:20])
    print()

# enumerate image files
for folder in sorted(os.listdir(root)):
    full = os.path.join(root, folder)
    if os.path.isdir(full):
        files = [f for f in os.listdir(full) if os.path.isfile(os.path.join(full,f))]
        if files:
            print(folder, 'files:', len(files), 'sample:', files[:5])
        else:
            print(folder, 'files: 0')
    

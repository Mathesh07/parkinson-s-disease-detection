import psutil

for p in psutil.process_iter(['pid', 'name', 'cmdline']):
    try:
        if p.info['name'] and p.info['name'].lower().startswith('python'):
            print(p.info['pid'], p.info['name'], p.info['cmdline'])
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        continue

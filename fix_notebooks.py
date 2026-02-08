import json
import glob

for file in glob.glob('projects/*.ipynb'):
    try:
        with open(file, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        
        # Remove problematic metadata
        if 'metadata' in nb and 'widgets' in nb['metadata']:
            del nb['metadata']['widgets']
        
        with open(file, 'w', encoding='utf-8') as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        
        print(f'Fixed: {file}')
    except Exception as e:
        print(f'Error in {file}: {e}')
import json
import glob
import os

for file in glob.glob('projects/*.ipynb'):
    try:
        with open(file, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        
        # Remove ALL widget metadata to fix corruption
        if 'metadata' in nb:
            if 'widgets' in nb['metadata']:
                del nb['metadata']['widgets']
            if 'language_info' in nb['metadata']:
                if isinstance(nb['metadata']['language_info'], dict):
                    nb['metadata']['language_info'].pop('codemirror_mode', None)
        
        # Fix cells metadata too
        if 'cells' in nb:
            for cell in nb['cells']:
                if 'metadata' in cell and 'widgets' in cell['metadata']:
                    del cell['metadata']['widgets']
        
        with open(file, 'w', encoding='utf-8') as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        
        print(f'✓ Fixed: {file}')
    except Exception as e:
        print(f'✗ Error in {file}: {e}')

print("\nAll notebooks fixed!")
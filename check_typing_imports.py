#!/usr/bin/env python3
"""
Script pour vérifier les imports typing manquants dans le codebase
"""

import os
import re

def check_file(filepath):
    """Vérifie si un fichier utilise Dict, List, etc. sans les importer"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Vérifier les imports typing
        has_typing_import = bool(re.search(r'^from typing import|^import typing', content, re.MULTILINE))

        # Vérifier l'utilisation de types typing
        uses_dict = bool(re.search(r':\s*Dict\[|-> Dict\[', content))
        uses_list = bool(re.search(r':\s*List\[|-> List\[', content))
        uses_tuple = bool(re.search(r':\s*Tuple\[|-> Tuple\[', content))
        uses_optional = bool(re.search(r':\s*Optional\[|-> Optional\[', content))
        uses_any = bool(re.search(r':\s*Any[,\s\]]|-> Any', content))

        uses_typing = uses_dict or uses_list or uses_tuple or uses_optional or uses_any

        if uses_typing and not has_typing_import:
            print(f"\n[MISSING] {filepath}")
            if uses_dict:
                print("    - Uses Dict but missing import")
            if uses_list:
                print("    - Uses List but missing import")
            if uses_tuple:
                print("    - Uses Tuple but missing import")
            if uses_optional:
                print("    - Uses Optional but missing import")
            if uses_any:
                print("    - Uses Any but missing import")
            return False

        return True

    except Exception as e:
        print(f"[ERROR] {filepath}: {e}")
        return True

def main():
    print("Checking typing imports in Python files...\n")
    print("=" * 70)

    issues_found = []

    for root, dirs, files in os.walk('.'):
        # Skip certain directories
        if any(skip in root for skip in ['.git', '__pycache__', 'venv', '.venv']):
            continue

        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                if not check_file(filepath):
                    issues_found.append(filepath)

    print("\n" + "=" * 70)

    if issues_found:
        print(f"\n[SUMMARY] Found {len(issues_found)} file(s) with missing typing imports:")
        for f in issues_found:
            print(f"  - {f}")
    else:
        print("\n[OK] All files have proper typing imports!")

    print()

if __name__ == "__main__":
    main()

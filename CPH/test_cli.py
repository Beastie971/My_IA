#!/usr/bin/env python3
import argparse
import sys

print("Script test démarré")

parser = argparse.ArgumentParser(description="Test CLI")
parser.add_argument('--list-models', action='store_true', help='Test')
args = parser.parse_args()

if args.list_models:
    print("Liste des modèles test")
else:
    print("Mode GUI test")

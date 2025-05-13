#!/usr/bin/env python3
import json
import os

# Load the dataset
with open('data/SWE-Bench-CL.json', 'r') as f:
    data = json.load(f)

# Collect unique difficulty values
difficulties = set()
for sequence in data.get('sequences', []):
    for task in sequence.get('tasks', []):
        difficulty = task.get('metadata', {}).get('difficulty', '')
        if difficulty:
            difficulties.add(difficulty)

# Print all unique difficulty values
print('Unique difficulty values in SWE-Bench-CL.json:')
for diff in sorted(difficulties):
    print(f'- "{diff}"')
print(f'\nTotal unique difficulties: {len(difficulties)}')

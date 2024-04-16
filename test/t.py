import pickle
import torch

with open("assignments.pkl", 'rb') as f:
    assignments = pickle.load(f)
b=list(assignments.values())
b.pop()

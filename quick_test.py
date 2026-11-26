"""Quick sanity test: load Karate, run DMRA-lite, print metrics."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import set_seed
from src.data import load_karate
from src.models import DMRALite, SymNMF
from src.evaluate import evaluate_all
from src.motif import count_triangles

set_seed(42)

print("Loading Karate Club...")
A, labels, name = load_karate()
n = A.shape[0]
k = len(set(labels))
print(f"  {name}: n={n}, edges={int(A.sum()//2)}, k={k}, triangles={count_triangles(A)}")

print("\nRunning SymNMF...")
symnmf = SymNMF(k=k, seed=42)
pred_sym = symnmf.fit(A, verbose=True)
m = evaluate_all(labels, pred_sym)
print(f"  SymNMF => ACC={m['ACC']:.3f} NMI={m['NMI']:.3f} ARI={m['ARI']:.3f} AMI={m['AMI']:.3f}")

for alpha in [0.0, 0.3, 0.5, 0.7]:
    print(f"\nRunning DMRA-lite (alpha={alpha})...")
    model = DMRALite(k=k, alpha=alpha, max_iter=300, seed=42, depth=2)
    pred = model.fit(A, verbose=True)
    m = evaluate_all(labels, pred)
    print(f"  alpha={alpha} => ACC={m['ACC']:.3f} NMI={m['NMI']:.3f} ARI={m['ARI']:.3f} AMI={m['AMI']:.3f}")

print("\n=== SANITY TEST COMPLETE ===")

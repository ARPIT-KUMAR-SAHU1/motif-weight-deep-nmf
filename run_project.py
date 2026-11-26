"""One-command entry point for the whole project.

    python run_project.py check       # diagnose the environment
    python run_project.py tests       # run the unit tests
    python run_project.py quick       # run quick experiments
    python run_project.py full        # run full experiments
    python run_project.py dashboard   # open the local dashboard
    python run_project.py all         # check -> tests -> quick run -> dashboard

If a command does nothing, run 'python run_project.py check' first.
"""
import importlib.util
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def run(args):
    print(f"\n>>> {sys.executable} {' '.join(args)}\n", flush=True)
    return subprocess.call([sys.executable] + args, cwd=ROOT)


def check():
    """Report exactly what is installed and which files are present."""
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"Project: {ROOT}")
    required = ["numpy", "scipy", "sklearn", "matplotlib", "pandas",
                "networkx", "torch", "flask"]
    for name in required:
        present = importlib.util.find_spec(name) is not None
        print(f"  {name:12s} {'OK' if present else 'MISSING (pip install -r requirements.txt)'}")
    optional = ["torch_geometric", "igraph", "leidenalg", "pytest"]
    for name in optional:
        present = importlib.util.find_spec(name) is not None
        print(f"  {name:16s} {'OK' if present else 'optional - fallbacks are used'}")
    for folder in ("src", "ui", "ui/templates", "ui/static", "experiments", "tests"):
        path = os.path.join(ROOT, folder)
        print(f"  {folder:16s} {'OK' if os.path.isdir(path) else 'MISSING'}")
    for path in ("run_dashboard.py", "ui/templates/index.html",
                 "results/summary.json"):
        print(f"  {path:26s} {'OK' if os.path.exists(os.path.join(ROOT, path)) else 'MISSING (run quick first)'}")
    print("\nTo start: python run_project.py all")
    return 0


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "check":
        return check()
    if cmd == "tests":
        return run(["-m", "unittest", "discover", "-s", "tests", "-v"])
    if cmd == "quick":
        return run(["experiments/run_all.py", "--quick"])
    if cmd == "full":
        return run(["experiments/run_all.py"])
    if cmd in ("dashboard", "ui"):
        return run(["run_dashboard.py"])
    if cmd == "all":
        print("Step 1/3: environment check")
        check()
        print("\nStep 2/3: unit tests")
        if run(["-m", "unittest", "discover", "-s", "tests"]) != 0:
            print("Tests failed - fix them before generating results.")
            return 1
        if not os.path.exists(os.path.join(ROOT, "results", "summary.json")):
            print("\nStep 3/3: generating results (first run only)")
            if run(["experiments/run_all.py", "--quick"]) != 0:
                return 1
        else:
            print("\nStep 3/3: existing results found - starting dashboard")
        return run(["run_dashboard.py"])
    print(f"Unknown command: {cmd}")
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())

"""Run pytest and save output to a UTF-8 file."""
import subprocess, sys, os

result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short", "--no-header"],
    capture_output=True,
    text=True,
    encoding="utf-8",
    cwd=os.path.dirname(os.path.abspath(__file__)),
)

output = result.stdout + result.stderr
with open("pytest_results.txt", "w", encoding="utf-8") as f:
    f.write(output)
    f.write(f"\n\nRETURN CODE: {result.returncode}\n")

print(output)
print("RETURN CODE:", result.returncode)

import subprocess, sys, os

result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short", "-m", "not slow"],
    capture_output=True,
    text=True,
    encoding="utf-8",
    errors="replace",
    cwd=os.path.dirname(os.path.abspath(__file__)),
)

output = result.stdout + result.stderr
with open("pytest_results.txt", "w", encoding="utf-8") as f:
    f.write(output)
    f.write(f"\nRETURN CODE: {result.returncode}\n")


lines = output.splitlines()
for line in lines:
    print(line)
print("RETURN CODE:", result.returncode)

import shutil
import subprocess

p = shutil.which("ollama")
if not p:
    print("OLLAMA_NOT_FOUND")
else:
    print(p)
    try:
        proc = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
        if proc.returncode == 0:
            print(proc.stdout.strip())
        else:
            print(proc.stderr.strip() or proc.stdout.strip())
    except Exception as e:
        print("VERSION_ERROR", str(e))

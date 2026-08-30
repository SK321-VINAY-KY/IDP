import importlib.util
libs = [
    "instructor", "openai", "tenacity", "requests",
    "paddleocr", "pymupdf", "PIL", "numpy",
    "pydantic_settings", "docling",
]
for lib in libs:
    found = importlib.util.find_spec(lib) is not None
    print(f"{'OK     ' if found else 'MISSING'} {lib}")

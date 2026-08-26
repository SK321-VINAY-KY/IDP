import importlib, json
names = ['google.genai', 'google.generativeai', 'google_generativeai']
out = {}
for name in names:
    try:
        m = importlib.import_module(name)
        out[name] = {'found': True, 'first_attrs': dir(m)[:30]}
    except Exception as e:
        out[name] = {'found': False, 'error': str(e)}
print(json.dumps(out, indent=2))

import sys
import os
try:
    # Try to source API key from our settings before importing the SDK
    try:
        from src.config.settings import settings
        if getattr(settings, 'gemini_api_key', None):
            os.environ.setdefault('GOOGLE_API_KEY', settings.gemini_api_key)
    except Exception:
        pass

    import google.generativeai as genai
except Exception as e:
    print('import_failed', type(e).__name__, str(e))
    sys.exit(0)

print('has_list_models', hasattr(genai, 'list_models'))
try:
    # Try to configure from our settings file (reads .env via pydantic-settings)
    try:
        from src.config.settings import settings
        if getattr(settings, 'gemini_api_key', None):
            genai.configure(api_key=settings.gemini_api_key)
    except Exception:
        pass

    it = genai.list_models()
    for m in it:
        print(repr(m))
except Exception as e:
    print('list_models_failed', type(e).__name__, str(e))

#!/usr/bin/env python3

print("🔍 Debug étape par étape...")

try:
    print("1. Import sys...")
    import sys
    print("✅ sys OK")
    
    print("2. Import os...")
    import os
    print("✅ os OK")
    
    print("3. Import config...")
    from config import check_dependencies, PROMPT_STORE_DIR
    print("✅ config OK")
    
    print("4. Check dependencies...")
    check_dependencies()
    print("✅ check_dependencies OK")
    
    print("5. Import ai_providers...")
    from ai_providers import get_ollama_models
    print("✅ ai_providers OK")
    
    print("6. Test get_ollama_models (peut bloquer ici)...")
    models = get_ollama_models()
    print(f"✅ Models: {len(models)}")
    
    print("7. Import gradio_interface...")
    from gradio_interface import build_ui
    print("✅ gradio_interface OK")
    
    print("8. Build UI...")
    app = build_ui()
    print("✅ UI built")
    
except Exception as e:
    print(f"❌ ERREUR: {e}")
    import traceback
    traceback.print_exc()

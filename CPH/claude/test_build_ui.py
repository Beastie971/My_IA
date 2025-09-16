#!/usr/bin/env python3

print("🔍 Test import...")

try:
    from gradio_interface import build_ui
    print("✅ Import réussi")
    
    print("🔍 Test build_ui()...")
    result = build_ui()
    print(f"✅ build_ui() retourne : {type(result)}")
    
    if result is None:
        print("❌ PROBLÈME : build_ui() retourne None !")
    else:
        print("✅ build_ui() retourne un objet valide")
        
except Exception as e:
    print(f"❌ ERREUR : {e}")
    import traceback
    traceback.print_exc()

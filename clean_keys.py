import os
import re
import shutil

# La clave detectada (puedes cambiarla si es otra en el futuro)
API_PATTERN = r'AIza[0-9A-Za-z\-_]{35}'

# Archivos donde se detectaron claves
FILES_TO_CLEAN = [
    "nextiadata.html",
    ".env",
    "app_backup.py",
    "backups/nextiadata.html.backup",
]

def clean_file(filepath):
    if not os.path.exists(filepath):
        print(f"⚠️  No encontrado: {filepath}")
        return

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    new_content = re.sub(API_PATTERN, "TU_NUEVA_API_KEY_AQUI", content)

    # Hacer backup antes de limpiar
    backup_path = filepath + ".bak"
    shutil.copy(filepath, backup_path)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"✅ Claves limpiadas en: {filepath} (backup en {backup_path})")

def remove_file(filepath):
    if os.path.exists(filepath):
        os.remove(filepath)
        print(f"🗑️  Eliminado: {filepath}")
    else:
        print(f"⚠️  No encontrado para borrar: {filepath}")

if __name__ == "__main__":
    print("🔍 Iniciando limpieza de claves...")
    for file in FILES_TO_CLEAN:
        if "backup" in file:
            remove_file(file)
        else:
            clean_file(file)
    print("✨ Limpieza completada.")

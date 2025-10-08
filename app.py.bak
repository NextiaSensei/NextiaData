# Creating secure app files and helper scripts for the user to download.
from pathlib import Path
import zipfile, os, textwrap, json

out_dir = Path("/mnt/data/nextia_secure_files")
out_dir.mkdir(parents=True, exist_ok=True)

# app_secure.py content (no keys included)
app_py = textwrap.dedent("""
    # app_secure.py - NextiaData (versión segura)
    from flask import Flask, request, jsonify, send_file, send_from_directory
    from flask_cors import CORS
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import io
    import base64
    import json
    import os
    from datetime import datetime
    from collections import deque

    # Cargar variables de entorno desde .env (opcional en desarrollo)
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        # python-dotenv no instalado o no deseado en producción
        pass

    plt.switch_backend('Agg')

    app = Flask(__name__)
    CORS(app)

    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    if not GEMINI_API_KEY:
        print("[WARN] GEMINI_API_KEY no encontrada en variables de entorno. Configura GEMINI_API_KEY en .env o en el entorno del sistema.")

    # Importar cliente de Google Generative AI sólo si está disponible
    try:
        import google.generativeai as genai
        if GEMINI_API_KEY:
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel('gemini-2.0-flash')
        else:
            model = None
    except Exception as e:
        print(f"[WARN] google.generativeai no disponible: {e}")
        model = None

    # Memoria simple en memoria (para producción usa Redis o base de datos)
    conversation_history = {}

    def get_conversation_history(user_id="default", max_messages=10):
        if user_id not in conversation_history:
            conversation_history[user_id] = deque(maxlen=max_messages)
        return conversation_history[user_id]

    # Rutas
    @app.route('/nextiadata.html')
    def serve_nextia_html():
        return send_from_directory(os.path.dirname(__file__), 'nextiadata.html')

    @app.route('/')
    def home():
        return "<h1>NextiaData Backend funcionando correctamente!</h1><p>Ve a <a href='/nextiadata.html'>NextiaData</a></p>"

    @app.route('/ai/ask', methods=['POST'])
    def ask_ai():
        try:
            data = request.get_json(force=True) or {}
            user_id = data.get('user_id', 'default')
            user_message = data.get('message', '')
            dataset_info = data.get('dataset_info', '')

            history = get_conversation_history(user_id)
            history.append(f"Usuario: {user_message}")

            prompt = (
                f"Eres NextiaAI, el asistente de IA para análisis de datos empresariales.\\n"
                f"CONTEXT: {''.join(list(history)[-3:]) if history else 'Primera interacción'}\\n"
                f"DATASET: {dataset_info}\\n"
                f"PREGUNTA: {user_message}\\n"
                "Proporciona insights accionables con métricas y recomendaciones."
            )

            if model is None:
                reply = ("El motor de IA no está configurado en este servidor. "
                         "Configura GEMINI_API_KEY en las variables de entorno para habilitar respuestas generadas.")
            else:
                response = model.generate_content(prompt)
                reply = getattr(response, 'text', str(response))

            history.append(f"Asistente: {reply}")

            return jsonify({"success": True, "response": reply, "user_id": user_id})
        except Exception as e:
            print(f"Error en /ai/ask: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    def _dataset_to_df(dataset):
        # Acepta dataset como {headers: [...], rows: [[...], ...]} o lista de dicts o lista de listas
        if dataset is None:
            return pd.DataFrame()
        if isinstance(dataset, dict) and 'headers' in dataset and 'rows' in dataset:
            try:
                return pd.DataFrame(dataset['rows'], columns=dataset['headers'])
            except Exception:
                # Fallback: try to create without columns
                return pd.DataFrame(dataset['rows'])
        else:
            return pd.DataFrame(dataset)

    @app.route('/analyze', methods=['POST'])
    def analyze_data():
        try:
            data = request.get_json(force=True) or {}
            df = _dataset_to_df(data.get('dataset'))
            analysis = {
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": list(df.columns),
                "summary": df.describe(include='all').to_dict(),
                "missing_values": df.isnull().sum().to_dict(),
                "data_types": {col: str(dtype) for col, dtype in df.dtypes.to_dict().items()}
            }
            return jsonify({"success": True, "analysis": analysis})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route('/visualize', methods=['POST'])
    def visualize_data():
        try:
            data = request.get_json(force=True) or {}
            df = _dataset_to_df(data.get('dataset'))
            chart_type = data.get('chart_type', 'line')
            x_col = int(data.get('x_column', 0))
            y_col = int(data.get('y_column', 1))

            plt.figure(figsize=(10, 6))

            # Try convert columns to numeric where possible for plotting
            try:
                if len(df.columns) > 0:
                    df_numeric = df.copy()
                    for c in df_numeric.columns:
                        df_numeric[c] = pd.to_numeric(df_numeric[c], errors='ignore')
                else:
                    df_numeric = df
            except Exception:
                df_numeric = df

            if chart_type == 'line' and len(df_numeric.columns) >= 2:
                plt.plot(df_numeric.iloc[:, x_col], df_numeric.iloc[:, y_col], marker='o', linewidth=2, markersize=4)
                plt.title(f'{df_numeric.columns[y_col]} vs {df_numeric.columns[x_col]}' if len(df_numeric.columns) > max(x_col,y_col) else 'Line chart')
            elif chart_type == 'bar' and len(df_numeric.columns) >= 2:
                plt.bar(df_numeric.iloc[:, x_col], df_numeric.iloc[:, y_col])
                plt.title('Bar chart')
            elif chart_type == 'histogram' and len(df_numeric.columns) >= 1:
                plt.hist(df_numeric.iloc[:, x_col].dropna().astype(float), bins=15, edgecolor='black')
                plt.title('Histogram')
            elif chart_type == 'scatter' and len(df_numeric.columns) >= 2:
                plt.scatter(df_numeric.iloc[:, x_col], df_numeric.iloc[:, y_col])
                plt.title('Scatter plot')
            else:
                df_numeric.plot()
                plt.title('Visualización de Datos')

            if len(df_numeric.columns) > x_col:
                plt.xlabel(str(df_numeric.columns[x_col]))
            if len(df_numeric.columns) > y_col:
                plt.ylabel(str(df_numeric.columns[y_col]))

            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            img = io.BytesIO()
            plt.savefig(img, format='png', dpi=100)
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()
            plt.close()

            return jsonify({"success": True, "image": plot_url})
        except Exception as e:
            print(f"Error en /visualize: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route('/export', methods=['POST'])
    def export_data():
        try:
            data = request.get_json(force=True) or {}
            df = _dataset_to_df(data.get('dataset'))

            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, sheet_name='Datos', index=False)
                df.describe(include='all').to_excel(writer, sheet_name='Análisis')
            output.seek(0)

            return send_file(
                output,
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                as_attachment=True,
                download_name=f'nextia_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
            )
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    if __name__ == '__main__':
        host = os.getenv('NEXTIA_HOST', '127.0.0.1')
        port = int(os.getenv('NEXTIA_PORT', 5000))
        debug = os.getenv('NEXTIA_DEBUG', 'True').lower() in ('1', 'true', 'yes')

        print("✨ Iniciando NextiaData Server (versión segura)...")
        app.run(debug=debug, host=host, port=port)
""").strip()

# remove_key.py - helper to remove the GEMINI key from nextiadata.html safely
remove_key_py = textwrap.dedent(r"""
    # remove_key.py -- elimina líneas que definen GEMINI_API_KEY o contienen 'AIza' dentro de un HTML/JS.
    import sys
    import re
    from pathlib import Path

    def backup(path: Path):
        bak = path.with_suffix(path.suffix + '.bak')
        path.rename(bak)
        return bak

    def clean_file(path: Path):
        text = path.read_text(encoding='utf-8')
        # Remove JS const GEMINI_API_KEY = "...";
        text_new = re.sub(r'const\s+GEMINI_API_KEY\s*=\s*["\'].*?["\'];?\s*\\n', '', text, flags=re.IGNORECASE)
        # Remove any bare API key patterns (simple heuristic for 'AIza' keys)
        text_new = re.sub(r'["\'](AIza[0-9A-Za-z_-]{35,})["\']', '"REDACTED_KEY_REMOVED"', text_new)
        return text_new

    def main():
        if len(sys.argv) < 2:
            print("Uso: python remove_key.py path/to/nextiadata.html")
            return
        p = Path(sys.argv[1])
        if not p.exists():
            print(f"No existe: {p}")
            return
        bak = p.with_suffix(p.suffix + '.bak')
        if not bak.exists():
            p.rename(bak)
        else:
            # if bak exists, create timestamped backup
            import datetime
            p.rename(p.with_name(p.stem + "_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + p.suffix))
        # read from backup and write cleaned file
        new_text = clean_file(bak)
        p.write_text(new_text, encoding='utf-8')
        print(f"Clave removida (si existía). Backup creado: {bak}")

    if __name__ == '__main__':
        main()
""").strip()

# patch.sh - bash script users can run (Linux/Mac) to backup and run cleaner
patch_sh = textwrap.dedent("""
    #!/usr/bin/env bash
    # patch.sh -- crea backup y limpia nextiadata.html de claves embebidas
    set -e
    FILE="nextiadata.html"
    if [ ! -f "$FILE" ]; then
      echo "No se encontró $FILE en el directorio actual. Coloca este script en la carpeta del proyecto."
      exit 1
    fi
    python3 remove_key.py "$FILE"
    echo "Limpieza completada. Revisa nextiadata.html y el archivo .bak"
""").strip()

# requirements.txt
requirements_txt = textwrap.dedent("""
    flask>=2.2
    flask-cors>=3.0
    pandas>=1.5
    numpy>=1.23
    matplotlib>=3.5
    python-dotenv>=0.21
    xlsxwriter>=3.0
    google-cloud-aiplatform>=1.29.0
    google-generative-ai>=0.1.0
""").strip()

# .env.example
env_example = textwrap.dedent("""
    # .env.example — renombra a .env y coloca tu clave (NO subir .env al repo)
    GEMINI_API_KEY=TU_NUEVA_API_KEY_AQUI
    NEXTIA_HOST=127.0.0.1
    NEXTIA_PORT=5000
    NEXTIA_DEBUG=True
""").strip()

# README quick into
readme_txt = textwrap.dedent("""
    README - NextiaData secure files
    --------------------------------
    Contenido de este paquete:
    - app_secure.py       -> versión segura de app.py (sin claves)
    - requirements.txt    -> dependencias sugeridas
    - .env.example        -> ejemplo de variables de entorno
    - remove_key.py       -> script para eliminar GEMINI API key del HTML
    - patch.sh            -> script bash para ejecutar remove_key.py
    --------------------------------
    Cómo usar:
    1) Copia estos archivos dentro de la carpeta del proyecto NextiaData.
    2) Ejecuta: python3 -m venv venv && source venv/bin/activate
    3) pip install -r requirements.txt
    4) cp .env.example .env  # luego edita .env con tu nueva key
    5) python3 app_secure.py
    6) Para limpiar HTML de keys embebidas: python3 remove_key.py nextiadata.html
    --------------------------------
    NOTA: No incluí ninguna API key real en estos archivos. Si tu repo ya tuvo una clave comprometida,
    revócala en Google Cloud Console y genera una nueva antes de ponerla en .env.
""").strip()

# Write files
files = {
    "app_secure.py": app_py,
    "remove_key.py": remove_key_py,
    "patch.sh": patch_sh,
    "requirements.txt": requirements_txt,
    ".env.example": env_example,
    "README_nextia_secure.txt": readme_txt
}

for name, content in files.items():
    p = out_dir / name
    p.write_text(content, encoding='utf-8')
    # Ensure script executable mode for patch.sh
    if name == "patch.sh":
        os.chmod(p, 0o755)

# Create zip
zip_path = Path("/mnt/data/nextia_secure_files.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for filepath in out_dir.iterdir():
        zf.write(filepath, arcname=filepath.name)

# Print files created
created = [str(p) for p in out_dir.iterdir()]
{"created_files": created, "zip": str(zip_path)}



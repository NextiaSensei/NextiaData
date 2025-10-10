import os
import pandas as pd
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from flask import Flask, request, jsonify, send_from_directory, send_file
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from flask_cors import CORS
import numpy as np
from datetime import datetime

# Cargar variables de entorno
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

print(f"[DEBUG] GEMINI_API_KEY cargada: {GEMINI_API_KEY is not None}")

# Inicializar modelo Gemini
model = None
try:
    import google.generativeai as genai
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.5-flash')
        print("[OK] Gemini 2.5 Flash configurado correctamente ✅")
    else:
        print("[ERROR] GEMINI_API_KEY no encontrada")
except Exception as e:
    print(f"[ERROR] Configuración Gemini: {e}")
    model = None

# Crear app Flask
app = Flask(__name__)
CORS(app)

# Configuración
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Estado de la aplicación
conversation_history = {}
user_datasets = {}

def get_conversation_history(user_id="default", max_messages=10):
    if user_id not in conversation_history:
        conversation_history[user_id] = []
    conversation_history[user_id] = conversation_history[user_id][-max_messages:]
    return conversation_history[user_id]

def get_ai_analysis(user_message, user_id="default"):
    user_data = user_datasets.get(user_id, {})
    
    if user_data and 'filename' in user_data:
        prompt = f"""
        Eres NextiaData Assistant, un experto en análisis de datos, ciencia de datos y marketing analytics. 
        Responde de manera DETALLADA, ÚTIL y COMPLETA.

        INFORMACIÓN DEL DATASET CARGADO:
        - Archivo: {user_data['filename']}
        - Dimensiones: {user_data['shape']} (filas, columnas)
        - Columnas disponibles: {user_data['columns']}
        - Columnas numéricas: {user_data['numeric_columns']}
        - Columnas categóricas: {user_data['categorical_columns']}
        - Muestra de datos: {user_data['sample'][:2] if user_data['sample'] else 'No disponible'}

        CONTEXTO Y MEMORIA:
        Historial reciente: {get_conversation_history(user_id)[-3:]}

        PREGUNTA DEL USUARIO: {user_message}

        INSTRUCCIONES ESPECÍFICAS:
        1. Proporciona un análisis COMPLETO y DETALLADO relacionado con la pregunta
        2. Incluye recomendaciones específicas de visualizaciones (qué gráficos hacer y por qué)
        3. Ofrece insights accionables para negocios/marketing
        4. Sugiere próximos pasos de análisis
        5. Si es relevante, menciona patrones, tendencias o correlaciones potenciales
        6. Sé exhaustivo pero claro, usando ejemplos concretos cuando sea posible

        Responde en español con un tono profesional pero accesible.
        """
    else:
        prompt = f"""
        Eres NextiaData Assistant, especialista en análisis de datos, machine learning y marketing analytics.
        
        PREGUNTA: {user_message}

        Proporciona una respuesta COMPLETA y DETALLADA. 
        Explica cómo puedes ayudar con análisis de datos, visualizaciones y insights.
        Incluye ejemplos específicos de tipos de análisis que podemos realizar.
        Anima al usuario a cargar un dataset CSV para comenzar el análisis.

        Sé exhaustivo en tu respuesta, mostrando todo tu conocimiento y capacidades.
        Responde en español.
        """

    return prompt

@app.route('/')
def index():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        html_path = os.path.join(current_dir, 'nextiadata.html')
        
        if os.path.exists(html_path):
            return send_file(html_path)
        else:
            return "<h1>Error: Archivo nextiadata.html no encontrado</h1>", 404
    except Exception as e:
        return f"<h1>Error interno</h1><p>{str(e)}</p>", 500

@app.route('/ai/ask', methods=['POST'])
def ask_ai():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No JSON data provided"}), 400
            
        user_id = data.get('user_id', 'default')
        user_message = data.get('message', '')

        if not user_message:
            return jsonify({"success": False, "error": "Empty message"}), 400

        print(f"[DEBUG] Mensaje recibido: {user_message}")

        history = get_conversation_history(user_id)
        history.append(f"Usuario: {user_message}")

        prompt = get_ai_analysis(user_message, user_id)

        if not model:
            reply = "🤖 Asistente NextiaData: Motor de IA no disponible temporalmente. Puedes usar las herramientas de análisis y gráficos."
            print("[WARN] Modelo Gemini no inicializado")
        else:
            try:
                response = model.generate_content(prompt)
                reply = response.text
                print(f"[DEBUG] Gemini respondió exitosamente")
            except Exception as gemini_error:
                print(f"[ERROR] Gemini API: {str(gemini_error)}")
                reply = "🤖 Asistente NextiaData: Estoy teniendo problemas de conexión. Mientras tanto, puedes usar las herramientas de análisis cargando un dataset CSV."

        history.append(f"Asistente: {reply}")
        
        return jsonify({
            "success": True, 
            "response": reply, 
            "user_id": user_id
        })
        
    except Exception as e:
        print(f"[ERROR] en /ai/ask: {str(e)}")
        return jsonify({
            "success": False, 
            "error": "Error interno del servidor"
        }), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No se subió ningún archivo'})
        
        file = request.files['file']
        user_id = request.form.get('user_id', 'default')
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No se seleccionó archivo'})
        
        if file and file.filename.endswith('.csv'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                df = pd.read_csv(filepath)
                
                # Análisis básico del dataset
                numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
                
                basic_info = {
                    'filename': filename,
                    'filepath': filepath,
                    'shape': df.shape,
                    'columns': df.columns.tolist(),
                    'numeric_columns': numeric_columns,
                    'categorical_columns': categorical_columns,
                    'sample': df.head(5).to_dict('records'),
                    'missing_values': df.isnull().sum().to_dict(),
                    'upload_time': datetime.now().isoformat()
                }
                
                user_datasets[user_id] = basic_info
                
                # Agregar mensaje al historial
                history = get_conversation_history(user_id)
                history.append(f"Sistema: Dataset '{filename}' cargado. {df.shape[0]} filas, {df.shape[1]} columnas")
                
                return jsonify({
                    'success': True, 
                    'data': basic_info,
                    'message': f'✅ Dataset cargado: {filename} ({df.shape[0]} filas, {df.shape[1]} columnas)'
                })
                
            except Exception as e:
                return jsonify({'success': False, 'error': f'Error leyendo CSV: {str(e)}'})
        else:
            return jsonify({'success': False, 'error': 'Solo se permiten archivos CSV'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# ========== ENDPOINTS PARA GRÁFICOS CON MATPLOTLIB ==========

@app.route('/api/visualization/line', methods=['POST'])
def line_chart():
    try:
        data = request.get_json()
        user_id = data.get('user_id', 'default')
        x_column = data.get('x_column')
        y_column = data.get('y_column')
        
        print(f"[GRAPH] Solicitando gráfico de líneas: {x_column} vs {y_column}")
        
        user_data = user_datasets.get(user_id)
        if not user_data:
            return jsonify({'success': False, 'error': 'No hay dataset cargado'})
        
        filepath = user_data['filepath']
        df = pd.read_csv(filepath)
        
        # Verificar que las columnas existen
        if x_column not in df.columns:
            return jsonify({'success': False, 'error': f'Columna X {x_column} no encontrada'})
        if y_column not in df.columns:
            return jsonify({'success': False, 'error': f'Columna Y {y_column} no encontrada'})
        
        # Limpiar datos
        df_clean = df.dropna(subset=[x_column, y_column])
        
        if len(df_clean) == 0:
            return jsonify({'success': False, 'error': 'No hay datos válidos para graficar'})
        
        # Crear gráfico de líneas
        plt.figure(figsize=(12, 6))
        
        # Ordenar por la columna X
        df_sorted = df_clean.sort_values(by=x_column)
        
        plt.plot(df_sorted[x_column], df_sorted[y_column], marker='o', linewidth=2, markersize=4, color='#667eea')
        plt.title(f'Tendencia: {y_column} vs {x_column}', fontsize=14, fontweight='bold')
        plt.xlabel(x_column, fontsize=12)
        plt.ylabel(y_column, fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Convertir a base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        print(f"[GRAPH] Gráfico de líneas generado exitosamente")
        
        return jsonify({
            'success': True, 
            'image': f'data:image/png;base64,{image_base64}',
            'type': 'line',
            'message': f'Gráfico de líneas generado: {y_column} vs {x_column}'
        })
        
    except Exception as e:
        print(f"[ERROR] En gráfico de líneas: {str(e)}")
        return jsonify({'success': False, 'error': f'Error generando gráfico: {str(e)}'})

@app.route('/api/visualization/bar', methods=['POST'])
def bar_chart():
    try:
        data = request.get_json()
        user_id = data.get('user_id', 'default')
        x_column = data.get('x_column')
        y_column = data.get('y_column')
        
        print(f"[GRAPH] Solicitando gráfico de barras: {x_column}, {y_column}")
        
        user_data = user_datasets.get(user_id)
        if not user_data:
            return jsonify({'success': False, 'error': 'No hay dataset cargado'})
        
        filepath = user_data['filepath']
        df = pd.read_csv(filepath)
        
        if x_column not in df.columns:
            return jsonify({'success': False, 'error': f'Columna {x_column} no encontrada'})
        
        # Limpiar datos
        df_clean = df.dropna(subset=[x_column])
        if y_column and y_column in df.columns:
            df_clean = df_clean.dropna(subset=[y_column])
        
        if len(df_clean) == 0:
            return jsonify({'success': False, 'error': 'No hay datos válidos para graficar'})
        
        # Crear gráfico de barras
        plt.figure(figsize=(12, 6))
        
        if y_column and y_column in df_clean.columns:
            # Gráfico de barras con valores
            df_grouped = df_clean.groupby(x_column)[y_column].mean().sort_values(ascending=False).head(15)
            colors = plt.cm.viridis(np.linspace(0, 1, len(df_grouped)))
            bars = plt.bar(df_grouped.index, df_grouped.values, color=colors)
            plt.title(f'Promedio de {y_column} por {x_column}', fontsize=14, fontweight='bold')
            plt.ylabel(y_column, fontsize=12)
        else:
            # Conteo de categorías
            value_counts = df_clean[x_column].value_counts().head(15)
            colors = plt.cm.plasma(np.linspace(0, 1, len(value_counts)))
            bars = plt.bar(value_counts.index, value_counts.values, color=colors)
            plt.title(f'Frecuencia de {x_column}', fontsize=14, fontweight='bold')
            plt.ylabel('Frecuencia', fontsize=12)
        
        plt.xlabel(x_column, fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Convertir a base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        print(f"[GRAPH] Gráfico de barras generado exitosamente")
        
        return jsonify({
            'success': True, 
            'image': f'data:image/png;base64,{image_base64}',
            'type': 'bar',
            'message': f'Gráfico de barras generado para {x_column}'
        })
        
    except Exception as e:
        print(f"[ERROR] En gráfico de barras: {str(e)}")
        return jsonify({'success': False, 'error': f'Error generando gráfico: {str(e)}'})

@app.route('/api/visualization/histogram', methods=['POST'])
def histogram():
    try:
        data = request.get_json()
        user_id = data.get('user_id', 'default')
        column = data.get('column')
        
        print(f"[GRAPH] Solicitando histograma: {column}")
        
        user_data = user_datasets.get(user_id)
        if not user_data:
            return jsonify({'success': False, 'error': 'No hay dataset cargado'})
        
        filepath = user_data['filepath']
        df = pd.read_csv(filepath)
        
        if column not in df.columns:
            return jsonify({'success': False, 'error': f'Columna {column} no encontrada'})
        
        # Limpiar datos
        df_clean = df.dropna(subset=[column])
        
        if len(df_clean) == 0:
            return jsonify({'success': False, 'error': 'No hay datos válidos para graficar'})
        
        # Crear histograma
        plt.figure(figsize=(12, 6))
        
        n, bins, patches = plt.hist(df_clean[column], bins=20, alpha=0.7, color='#4CAF50', edgecolor='black')
        plt.title(f'Distribución de {column}', fontsize=14, fontweight='bold')
        plt.xlabel(column, fontsize=12)
        plt.ylabel('Frecuencia', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Convertir a base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        print(f"[GRAPH] Histograma generado exitosamente")
        
        return jsonify({
            'success': True, 
            'image': f'data:image/png;base64,{image_base64}',
            'type': 'histogram',
            'message': f'Histograma generado para {column}'
        })
        
    except Exception as e:
        print(f"[ERROR] En histograma: {str(e)}")
        return jsonify({'success': False, 'error': f'Error generando histograma: {str(e)}'})

# ========== ENDPOINT CRÍTICO: CHART.JS DATA ==========

@app.route('/api/chartjs/data', methods=['POST'])
def chartjs_data():
    """Endpoint específico para Chart.js que el frontend está solicitando"""
    try:
        data = request.get_json()
        user_id = data.get('user_id', 'default')
        chart_type = data.get('chart_type')
        x_column = data.get('x_column')
        y_column = data.get('y_column')
        
        print(f"[CHARTJS] Solicitando datos para {chart_type}: {x_column}, {y_column}")
        
        user_data = user_datasets.get(user_id)
        if not user_data:
            return jsonify({'success': False, 'error': 'No hay dataset cargado'})
        
        filepath = user_data['filepath']
        df = pd.read_csv(filepath)
        
        # Verificar columnas
        if x_column not in df.columns:
            return jsonify({'success': False, 'error': f'Columna X {x_column} no encontrada'})
        if y_column and y_column not in df.columns:
            return jsonify({'success': False, 'error': f'Columna Y {y_column} no encontrada'})
        
        # Limpiar datos
        df_clean = df.dropna(subset=[x_column])
        if y_column:
            df_clean = df_clean.dropna(subset=[y_column])
        
        if len(df_clean) == 0:
            return jsonify({'success': False, 'error': 'No hay datos válidos para graficar'})
        
        chart_data = {
            'success': True,
            'chart_type': chart_type,
            'data': None,
            'options': None
        }
        
        if chart_type == 'line':
            # Ordenar por X para línea continua
            df_sorted = df_clean.sort_values(by=x_column)
            chart_data['data'] = {
                'labels': df_sorted[x_column].astype(str).tolist(),
                'datasets': [{
                    'label': y_column,
                    'data': df_sorted[y_column].tolist(),
                    'borderColor': 'rgb(75, 192, 192)',
                    'backgroundColor': 'rgba(75, 192, 192, 0.2)',
                    'tension': 0.1,
                    'fill': True
                }]
            }
            chart_data['options'] = {
                'responsive': True,
                'plugins': {
                    'title': {
                        'display': True,
                        'text': f'Tendencia de {y_column}'
                    }
                },
                'scales': {
                    'x': {
                        'title': {
                            'display': True,
                            'text': x_column
                        }
                    },
                    'y': {
                        'title': {
                            'display': True,
                            'text': y_column
                        }
                    }
                }
            }
            
        elif chart_type == 'bar':
            if y_column:
                # Gráfico de barras con valores
                df_grouped = df_clean.groupby(x_column)[y_column].mean().sort_values(ascending=False).head(10)
                chart_data['data'] = {
                    'labels': df_grouped.index.tolist(),
                    'datasets': [{
                        'label': f'Promedio de {y_column}',
                        'data': df_grouped.values.tolist(),
                        'backgroundColor': 'rgba(54, 162, 235, 0.8)',
                        'borderColor': 'rgb(54, 162, 235)',
                        'borderWidth': 1
                    }]
                }
            else:
                # Conteo de categorías
                value_counts = df_clean[x_column].value_counts().head(10)
                chart_data['data'] = {
                    'labels': value_counts.index.tolist(),
                    'datasets': [{
                        'label': f'Frecuencia de {x_column}',
                        'data': value_counts.values.tolist(),
                        'backgroundColor': 'rgba(255, 99, 132, 0.8)',
                        'borderColor': 'rgb(255, 99, 132)',
                        'borderWidth': 1
                    }]
                }
            
            chart_data['options'] = {
                'responsive': True,
                'plugins': {
                    'title': {
                        'display': True,
                        'text': f'Gráfico de Barras - {x_column}'
                    }
                },
                'scales': {
                    'y': {
                        'beginAtZero': True
                    }
                }
            }
            
        elif chart_type == 'histogram':
            # Para histograma en Chart.js, usamos un gráfico de barras
            values = df_clean[x_column].tolist()
            chart_data['data'] = {
                'labels': [f'Bin {i+1}' for i in range(len(values))],
                'datasets': [{
                    'label': f'Distribución de {x_column}',
                    'data': values,
                    'backgroundColor': 'rgba(255, 159, 64, 0.8)',
                    'borderColor': 'rgb(255, 159, 64)',
                    'borderWidth': 1
                }]
            }
            chart_data['options'] = {
                'responsive': True,
                'plugins': {
                    'title': {
                        'display': True,
                        'text': f'Histograma de {x_column}'
                    }
                },
                'scales': {
                    'y': {
                        'beginAtZero': True
                    }
                }
            }
            
        else:
            return jsonify({'success': False, 'error': 'Tipo de gráfico no soportado'})
        
        print(f"[CHARTJS] Datos generados exitosamente para {chart_type}")
        return jsonify(chart_data)
        
    except Exception as e:
        print(f"[ERROR] En Chart.js: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

# ========== ENDPOINTS PARA ANÁLISIS ==========

@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        user_id = data.get('user_id', 'default')
        
        user_data = user_datasets.get(user_id)
        if not user_data:
            return jsonify({'success': False, 'error': 'No hay dataset cargado'})
        
        filepath = user_data['filepath']
        df = pd.read_csv(filepath)
        
        # Análisis básico mejorado
        analysis = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'data_types': df.dtypes.astype(str).to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
            'numeric_stats': df.describe().to_dict() if not df.select_dtypes(include=[np.number]).empty else {},
            'sample_data': df.head(3).to_dict('records')
        }
        
        return jsonify({'success': True, 'analysis': analysis})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/analyze/basic', methods=['POST'])
def analyze_basic():
    return analyze()

@app.route('/api/dataset/columns', methods=['GET'])
def get_dataset_columns():
    try:
        user_id = request.args.get('user_id', 'default')
        user_data = user_datasets.get(user_id)
        
        if not user_data:
            return jsonify({'success': False, 'error': 'No hay dataset cargado'})
            
        return jsonify({
            'success': True,
            'columns': user_data['columns'],
            'numeric_columns': user_data['numeric_columns'],
            'categorical_columns': user_data['categorical_columns']
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/generate_report', methods=['POST'])
def generate_report():
    try:
        user_id = request.get_json().get('user_id', 'default')
        user_data = user_datasets.get(user_id)
        
        if not user_data:
            return jsonify({'success': False, 'error': 'No hay dataset cargado'})
        
        # Reporte básico en HTML
        report_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Reporte NextiaData - {user_data['filename']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #667eea; color: white; padding: 20px; border-radius: 10px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 8px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📊 Reporte NextiaData</h1>
                <p>Archivo: {user_data['filename']} | Filas: {user_data['shape'][0]} | Columnas: {user_data['shape'][1]}</p>
            </div>
            <div class="section">
                <h2>Resumen del Dataset</h2>
                <p>Generado automáticamente por NextiaData Assistant</p>
            </div>
        </body>
        </html>
        """
        
        return jsonify({
            'success': True, 
            'report_html': report_html,
            'message': 'Reporte generado exitosamente'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Servir archivos estáticos
@app.route('/<path:filename>')
def serve_static(filename):
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return send_from_directory(current_dir, filename)
    except Exception as e:
        return "Archivo no encontrado", 404

if __name__ == '__main__':
    host = os.getenv('NEXTIA_HOST', '127.0.0.1')
    port = int(os.getenv('NEXTIA_PORT', 5000))
    debug = os.getenv('NEXTIA_DEBUG', 'True').lower() in ('1','true','yes')
    
    print(f"🚀 NextiaData iniciando en http://{host}:{port}")
    print("📊 Características activadas:")
    print("   ✅ Gemini 2.5 Flash - IA Inteligente")
    print("   ✅ Gráficos Interactivos (Líneas, Barras, Histogramas)")
    print("   ✅ Chart.js - Gráficos dinámicos")
    print("   ✅ Análisis Exploratorio")
    print("   ✅ Múltiples endpoints para compatibilidad")
    print("   ✅ Logs detallados para debugging")
    
    app.run(host=host, port=port, debug=debug)

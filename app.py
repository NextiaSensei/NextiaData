import os
import pandas as pd
import json
import matplotlib
matplotlib.use('Agg')  # Para evitar problemas con el backend gráfico
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from flask_cors import CORS
import numpy as np
from datetime import datetime

# Cargar variables de entorno
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Imprimir para depuración (pero ten cuidado de no exponer la clave en producción)
print(f"[DEBUG] GEMINI_API_KEY cargada: {GEMINI_API_KEY is not None}")

# Inicializar modelo
model = None
try:
    import google.generativeai as genai
except ModuleNotFoundError:
    print("[ERROR] google-generativeai no instalado. Ejecuta: pip install google-generativeai")
else:
    if GEMINI_API_KEY:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel('gemini-2.5-flash')
            print("[OK] Gemini configurado correctamente ✅")
        except Exception as e:
            print(f"[ERROR] Falló configurar Gemini: {e}")
    else:
        print("[WARN] GEMINI_API_KEY no encontrada en .env")

# Crear app Flask
app = Flask(__name__, static_folder='.', template_folder='.')
CORS(app)  # Habilitar CORS para integración web

# Configurar carpeta para uploads
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

conversation_history = {}
user_datasets = {}

def get_conversation_history(user_id="default", max_messages=10):
    if user_id not in conversation_history:
        conversation_history[user_id] = []
    # Limitar el historial al número máximo de mensajes
    conversation_history[user_id] = conversation_history[user_id][-max_messages:]
    return conversation_history[user_id]

def get_ai_analysis(user_message, user_id="default"):
    """Mejorar el prompt para análisis de datos específicos"""
    user_data = user_datasets.get(user_id, {})
    
    if user_data and 'filename' in user_data:
        # Si hay dataset cargado, hacer análisis específico
        prompt = f"""
        Eres NextiaData Assistant, un experto en análisis de datos, ciencia de datos y marketing analytics.

        INFORMACIÓN DEL DATASET:
        - Archivo: {user_data['filename']}
        - Dimensiones: {user_data['shape']} (filas, columnas)
        - Columnas disponibles: {user_data['columns']}
        - Tipos de datos: {user_data['dtypes']}
        - Muestra de datos: {user_data['sample'][:3] if user_data['sample'] else 'No disponible'}

        CONTEXTO ACTUAL:
        Historial reciente: {get_conversation_history(user_id)[-3:]}
        
        PREGUNTA DEL USUARIO: {user_message}

        PROPORCIONA:
        1. Análisis específico relacionado con la pregunta
        2. Recomendaciones de visualizaciones apropiadas
        3. Insights accionables para negocios/marketing
        4. Posibles próximos pasos de análisis

        Sé conciso pero útil, enfocado en datos y resultados medibles.
        Si el dataset tiene columnas de fecha, sugiere análisis temporal.
        Si tiene columnas numéricas, sugiere correlaciones y estadísticas.
        """
    else:
        # Prompt general sin dataset
        prompt = f"""
        Eres NextiaData Assistant, especialista en análisis de datos, machine learning y marketing analytics.
        
        Historial: {get_conversation_history(user_id)[-3:]}
        Pregunta: {user_message}

        Responde de manera profesional y útil. Si el usuario quiere analizar datos, sugiérele que cargue un archivo CSV.
        Incluye ejemplos prácticos de cómo puedes ayudar con análisis de datos.
        """

    return prompt

def create_visualization(df, chart_type, x_column, y_column=None):
    """Crear visualizaciones y devolver como base64"""
    try:
        plt.figure(figsize=(10, 6))
        
        if chart_type == 'histogram' and x_column in df.columns:
            plt.hist(df[x_column].dropna(), bins=20, alpha=0.7, color='skyblue')
            plt.title(f'Distribución de {x_column}')
            plt.xlabel(x_column)
            plt.ylabel('Frecuencia')
            
        elif chart_type == 'scatter' and x_column in df.columns and y_column in df.columns:
            plt.scatter(df[x_column], df[y_column], alpha=0.6)
            plt.title(f'{y_column} vs {x_column}')
            plt.xlabel(x_column)
            plt.ylabel(y_column)
            
        elif chart_type == 'line' and x_column in df.columns and y_column in df.columns:
            # Intentar ordenar por la columna x si es numérica o de fecha
            temp_df = df.sort_values(by=x_column)
            plt.plot(temp_df[x_column], temp_df[y_column], marker='o')
            plt.title(f'Tendencia de {y_column} por {x_column}')
            plt.xlabel(x_column)
            plt.ylabel(y_column)
            plt.xticks(rotation=45)
            
        elif chart_type == 'bar' and x_column in df.columns:
            if y_column and y_column in df.columns:
                # Gráfico de barras con valores
                df_grouped = df.groupby(x_column)[y_column].mean().sort_values(ascending=False).head(10)
                df_grouped.plot(kind='bar', color='lightcoral')
                plt.title(f'Promedio de {y_column} por {x_column}')
                plt.ylabel(y_column)
            else:
                # Conteo de categorías
                df[x_column].value_counts().head(10).plot(kind='bar', color='lightgreen')
                plt.title(f'Frecuencia de {x_column}')
                plt.ylabel('Frecuencia')
            plt.xticks(rotation=45)
            
        elif chart_type == 'box' and x_column in df.columns:
            if y_column and y_column in df.columns:
                df.boxplot(column=y_column, by=x_column)
                plt.title(f'Distribución de {y_column} por {x_column}')
            else:
                df[x_column].plot(kind='box')
                plt.title(f'Diagrama de Caja de {x_column}')
            plt.xticks(rotation=45)
            
        else:
            return None
            
        plt.tight_layout()
        
        # Convertir gráfico a base64
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=100)
        img.seek(0)
        plt.close()
        
        return base64.b64encode(img.getvalue()).decode('utf-8')
        
    except Exception as e:
        print(f"Error creando visualización: {e}")
        plt.close()
        return None

@app.route('/')
def index():
    return send_from_directory(os.getcwd(), 'nextiadata.html')

@app.route('/ai/ask', methods=['POST'])
def ask_ai():
    try:
        data = request.get_json(force=True) or {}
        user_id = data.get('user_id', 'default')
        user_message = data.get('message', '')

        history = get_conversation_history(user_id)
        history.append(f"Usuario: {user_message}")

        # Usar el prompt mejorado
        prompt = get_ai_analysis(user_message, user_id)

        if not model:
            reply = "Motor de IA no configurado. Configura GEMINI_API_KEY en .env si quieres usar IA."
        else:
            response = model.generate_content(prompt)
            reply = getattr(response, 'text', str(response))

        history.append(f"Asistente: {reply}")
        
        # Incluir información del dataset si está cargado
        user_data = user_datasets.get(user_id, {})
        has_dataset = bool(user_data and 'filename' in user_data)
        
        return jsonify({
            "success": True, 
            "response": reply, 
            "user_id": user_id,
            "has_dataset": has_dataset
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        user_id = request.form.get('user_id', 'default')
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        if file and file.filename.endswith('.csv'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Leer el CSV para análisis básico
            try:
                df = pd.read_csv(filepath)
                
                # Análisis automático mejorado
                numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
                date_columns = []
                
                # Detectar columnas de fecha
                for col in df.columns:
                    if df[col].dtype == 'object':
                        try:
                            pd.to_datetime(df[col])
                            date_columns.append(col)
                        except:
                            pass
                
                basic_info = {
                    'filename': filename,
                    'filepath': filepath,
                    'shape': df.shape,
                    'columns': df.columns.tolist(),
                    'dtypes': df.dtypes.astype(str).to_dict(),
                    'sample': df.head(5).to_dict('records'),
                    'numeric_columns': numeric_columns,
                    'categorical_columns': categorical_columns,
                    'date_columns': date_columns,
                    'missing_values': df.isnull().sum().to_dict(),
                    'upload_time': datetime.now().isoformat()
                }
                
                # Guardar dataset para el usuario
                user_datasets[user_id] = basic_info
                
                # Agregar mensaje al historial
                history = get_conversation_history(user_id)
                history.append(f"Sistema: Dataset '{filename}' cargado exitosamente. {df.shape[0]} filas, {df.shape[1]} columnas")
                
                return jsonify({
                    'success': True, 
                    'data': basic_info,
                    'message': f'Dataset cargado: {filename} ({df.shape[0]} filas, {df.shape[1]} columnas)'
                })
                
            except Exception as e:
                return jsonify({'success': False, 'error': f'Error leyendo CSV: {str(e)}'})
        else:
            return jsonify({'success': False, 'error': 'Por favor sube un archivo CSV válido'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/analyze', methods=['POST'])
def analyze_data():
    try:
        data = request.get_json()
        user_id = data.get('user_id', 'default')
        analysis_type = data.get('analysis_type', 'basic')
        
        user_data = user_datasets.get(user_id)
        if not user_data:
            return jsonify({'success': False, 'error': 'No dataset loaded for user'})
        
        filepath = user_data['filepath']
        df = pd.read_csv(filepath)
        
        if analysis_type == 'basic':
            # Análisis básico mejorado
            describe_stats = df.describe(include='all').to_dict()
            
            # Información de correlaciones
            numeric_df = df.select_dtypes(include=[np.number])
            correlations = {}
            if not numeric_df.empty and numeric_df.shape[1] > 1:
                correlations = numeric_df.corr().to_dict()
            
            result = {
                'summary': describe_stats,
                'missing_values': df.isnull().sum().to_dict(),
                'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
                'correlations': correlations,
                'column_types': {
                    'numeric': user_data['numeric_columns'],
                    'categorical': user_data['categorical_columns'],
                    'date': user_data['date_columns']
                }
            }
            
        elif analysis_type == 'column_stats':
            column = data.get('column')
            if column not in df.columns:
                return jsonify({'success': False, 'error': f'Column {column} not found'})
            
            column_data = df[column]
            stats = column_data.describe().to_dict()
            
            # Estadísticas adicionales según el tipo de dato
            if np.issubdtype(column_data.dtype, np.number):
                additional_stats = {
                    'variance': column_data.var(),
                    'skewness': column_data.skew(),
                    'kurtosis': column_data.kurtosis(),
                    'outliers': len(column_data[(column_data - column_data.mean()).abs() > 3 * column_data.std()])
                }
            else:
                additional_stats = {
                    'unique_count': column_data.nunique(),
                    'most_frequent': column_data.mode().iloc[0] if not column_data.mode().empty else None,
                    'top_categories': column_data.value_counts().head(5).to_dict()
                }
            
            result = {
                'column': column,
                'stats': {**stats, **additional_stats},
                'unique_values': column_data.value_counts().head(10).to_dict() if column_data.dtype == 'object' else None,
                'data_type': str(column_data.dtype)
            }
            
        elif analysis_type == 'quick_insights':
            # Insights automáticos
            numeric_cols = user_data['numeric_columns']
            insights = []
            
            if len(numeric_cols) >= 2:
                # Encontrar correlaciones fuertes
                corr_matrix = df[numeric_cols].corr()
                strong_corrs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        if abs(corr_matrix.iloc[i, j]) > 0.7:
                            strong_corrs.append({
                                'columns': [corr_matrix.columns[i], corr_matrix.columns[j]],
                                'correlation': round(corr_matrix.iloc[i, j], 3)
                            })
                
                if strong_corrs:
                    insights.append({
                        'type': 'strong_correlation',
                        'message': 'Se encontraron correlaciones fuertes entre variables',
                        'data': strong_corrs
                    })
            
            result = {'insights': insights}
            
        else:
            return jsonify({'success': False, 'error': 'Invalid analysis type'})
        
        return jsonify({'success': True, 'analysis': result})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/visualize', methods=['POST'])
def create_visualization_endpoint():
    try:
        data = request.get_json()
        user_id = data.get('user_id', 'default')
        chart_type = data.get('chart_type')
        x_column = data.get('x_column')
        y_column = data.get('y_column')
        
        user_data = user_datasets.get(user_id)
        if not user_data:
            return jsonify({'success': False, 'error': 'No dataset loaded for user'})
        
        filepath = user_data['filepath']
        df = pd.read_csv(filepath)
        
        image_b64 = create_visualization(df, chart_type, x_column, y_column)
        
        if image_b64:
            return jsonify({
                'success': True, 
                'image': f"data:image/png;base64,{image_b64}",
                'chart_type': chart_type
            })
        else:
            return jsonify({'success': False, 'error': 'Could not create visualization'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/dataset_info', methods=['GET'])
def get_dataset_info():
    try:
        user_id = request.args.get('user_id', 'default')
        user_data = user_datasets.get(user_id)
        
        if user_data:
            return jsonify({'success': True, 'data': user_data})
        else:
            return jsonify({'success': False, 'error': 'No dataset loaded'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/clear_history', methods=['POST'])
def clear_history():
    try:
        data = request.get_json() or {}
        user_id = data.get('user_id', 'default')
        
        if user_id in conversation_history:
            conversation_history[user_id] = []
        
        return jsonify({'success': True, 'message': 'Historial limpiado'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/suggest_visualizations', methods=['GET'])
def suggest_visualizations():
    try:
        user_id = request.args.get('user_id', 'default')
        user_data = user_datasets.get(user_id)
        
        if not user_data:
            return jsonify({'success': False, 'error': 'No dataset loaded'})
        
        suggestions = []
        numeric_cols = user_data['numeric_columns']
        categorical_cols = user_data['categorical_columns']
        
        # Sugerencias basadas en los tipos de columnas
        if len(numeric_cols) >= 1:
            suggestions.append({
                'type': 'histogram',
                'name': 'Distribución',
                'description': f'Ver distribución de {numeric_cols[0]}',
                'x_column': numeric_cols[0]
            })
        
        if len(numeric_cols) >= 2:
            suggestions.append({
                'type': 'scatter',
                'name': 'Correlación',
                'description': f'Relación entre {numeric_cols[0]} y {numeric_cols[1]}',
                'x_column': numeric_cols[0],
                'y_column': numeric_cols[1]
            })
        
        if len(categorical_cols) >= 1:
            suggestions.append({
                'type': 'bar',
                'name': 'Frecuencia',
                'description': f'Conteo de categorías en {categorical_cols[0]}',
                'x_column': categorical_cols[0]
            })
        
        return jsonify({'success': True, 'suggestions': suggestions})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    host = os.getenv('NEXTIA_HOST', '127.0.0.1')
    port = int(os.getenv('NEXTIA_PORT', 5000))
    debug = os.getenv('NEXTIA_DEBUG', 'True').lower() in ('1','true','yes')
    print(f"✨ NextiaData arrancando en http://{host}:{port} (debug={debug})")
    print(f"📁 Upload folder: {app.config['UPLOAD_FOLDER']}")
    print("🚀 Características activadas: Uploads, Análisis, Visualizaciones, IA")
    app.run(host=host, port=port, debug=debug)

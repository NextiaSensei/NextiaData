# Importaciones COMPLETAS
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
import google.generativeai as genai

# Configuración de matplotlib
plt.switch_backend('Agg')

app = Flask(__name__)
CORS(app)

# ✅ CONFIGURACIÓN DE GEMINI AI (TU API KEY)
genai.configure(api_key="AIzaSyArn3TOT6P29eelnRo2TVYCB4bnAhtdISE")
model = genai.GenerativeModel('gemini-pro')

# ✅ Ruta para servir nextiadata.html
@app.route('/nextiadata.html')
def serve_nextia_html():
    return send_from_directory('.', 'nextiadata.html')

# ✅ Ruta principal
@app.route('/')
def home():
    return """
    <h1>NextiaData Backend funcionando correctamente!</h1>
    <p>El servidor Flask está ejecutándose.</p>
    <p>Ve a <a href='/nextiadata.html'>NextiaData</a> para usar la interfaz.</p>
    """

# ✅ Ruta para IA avanzada con Gemini
@app.route('/ai/ask', methods=['POST'])
def ask_ai():
    try:
        data = request.json
        user_message = data.get('message', '')
        dataset_info = data.get('dataset_info', '')
        
        # Crear prompt contextualizado para múltiples áreas
        prompt = f"""
        Eres NextiaAI, un asistente especializado en análisis de datos para múltiples áreas: 
        marketing, genética, ventas, investigación científica, negocios, etc.
        
        Contexto del usuario: {dataset_info}
        
        Pregunta: {user_message}
        
        Proporciona una respuesta técnica pero clara, con insights prácticos y recomendaciones 
        específicas para el área que corresponda. Si es análisis de datos, sugiere metodologías,
        visualizaciones adecuadas, y posibles insights.
        
        Responde en español de manera profesional pero accesible.
        """
        
        # Obtener respuesta de Gemini
        response = model.generate_content(prompt)
        
        return jsonify({
            "success": True, 
            "response": response.text,
            "type": "ai_response"
        })
        
    except Exception as e:
        print(f"Error en IA: {str(e)}")
        return jsonify({"success": False, "error": str(e)})

# ✅ Análisis de datos
@app.route('/analyze', methods=['POST'])
def analyze_data():
    try:
        data = request.json
        df = pd.DataFrame(data['dataset'])
        
        # Análisis básico
        analysis = {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "summary": df.describe().to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "data_types": {col: str(dtype) for col, dtype in df.dtypes.to_dict().items()}
        }
        
        return jsonify({"success": True, "analysis": analysis})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# ✅ Visualización de datos
@app.route('/visualize', methods=['POST'])
def visualize_data():
    try:
        data = request.json
        df = pd.DataFrame(data['dataset'])
        chart_type = data.get('chart_type', 'line')
        x_col = data.get('x_column', 0)
        y_col = data.get('y_column', 1)
        
        # Configurar tamaño y estilo
        plt.figure(figsize=(10, 6))
        plt.style.use('default')
        
        # Generar gráfico según el tipo seleccionado
        if chart_type == 'line' and len(df.columns) >= 2:
            plt.plot(df.iloc[:, x_col], df.iloc[:, y_col], marker='o', linewidth=2, markersize=4)
            plt.title(f'{df.columns[y_col]} vs {df.columns[x_col]}')
        elif chart_type == 'bar' and len(df.columns) >= 2:
            plt.bar(df.iloc[:, x_col], df.iloc[:, y_col])
            plt.title(f'{df.columns[y_col]} vs {df.columns[x_col]}')
        elif chart_type == 'histogram' and len(df.columns) >= 1:
            plt.hist(df.iloc[:, x_col], bins=15, alpha=0.7, edgecolor='black')
            plt.title(f'Distribución de {df.columns[x_col]}')
        elif chart_type == 'scatter' and len(df.columns) >= 2:
            plt.scatter(df.iloc[:, x_col], df.iloc[:, y_col])
            plt.title(f'{df.columns[y_col]} vs {df.columns[x_col]}')
        else:
            # Gráfico por defecto
            df.plot()
            plt.title('Visualización de Datos')
        
        # Añadir etiquetas
        if len(df.columns) > x_col:
            plt.xlabel(df.columns[x_col])
        if len(df.columns) > y_col:
            plt.ylabel(df.columns[y_col])
            
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Convertir imagen a base64
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=100)
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        
        # Limpiar la figura para liberar memoria
        plt.close()
        
        return jsonify({"success": True, "image": plot_url})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# ✅ Exportación de datos
@app.route('/export', methods=['POST'])
def export_data():
    try:
        data = request.json
        df = pd.DataFrame(data['dataset'])
        
        # Crear un archivo en memoria
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Datos', index=False)
            
            # Añadir análisis descriptivo
            df.describe().to_excel(writer, sheet_name='Análisis')
        
        output.seek(0)
        
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=f'nextia_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# ✅ Ejecución principal
if __name__ == '__main__':
    print("✨ Iniciando NextiaData Server con IA avanzada...")
    print("🌐 Servidor disponible en: http://127.0.0.1:5000")
    print("📊 Interfaz disponible en: http://127.0.0.1:5000/nextiadata.html")
    print("🤖 IA Gemini integrada y funcionando")
    app.run(debug=True, host='127.0.0.1', port=5000)

# NextiaData 🤖

**Asistente Virtual de IA para Ciencia de Datos y Marketing Analytics**

![NextiaData](https://img.shields.io/badge/NextiaData-AI%20Assistant-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![Flask](https://img.shields.io/badge/Flask-2.3.3-lightgrey)

## 🚀 Características

- **Análisis Inteligente de Datos** con Gemini AI
- **Visualizaciones Interactivas** en tiempo real
- **Carga de Datos CSV** con análisis automático
- **Chat Conversacional** para insights de datos
- **Reportes Automatizados** y exportación
- **Interfaz Moderna** y responsive

## 📦 Instalación Rápida

```bash
# Clonar repositorio
git clone https://github.com/JorgeSensei/NextiaData.git
cd NextiaData

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
cp .env.example .env
# Editar .env con tu GEMINI_API_KEY

# Ejecutar aplicación
python app.py

🛠️ Configuración
Obtén tu API key de Google AI Studio

Configura en .env:
GEMINI_API_KEY=tu_api_key_aqui
NEXTIA_HOST=127.0.0.1
NEXTIA_PORT=5000
NEXTIA_DEBUG=True

📊 Uso
Cargar Dataset: Sube tu archivo CSV

Analizar: Usa las herramientas de análisis

Visualizar: Genera gráficos automáticamente

Consultar: Chatea con el asistente IA

Exportar: Genera reportes PDF

🏗️ Estructura del Proyecto

NextiaData/
├── app.py                 # Aplicación principal Flask
├── nextiadata.html       # Frontend mejorado
├── requirements.txt      # Dependencias Python
├── .env.example         # Ejemplo de variables de entorno
├── uploads/             # Carpeta para archivos subidos
├── static/              # Recursos estáticos (opcional)
└── README.md           # Este archivo
🤝 Contribución
¡Contribuciones son bienvenidas! Por favor:

Fork el proyecto

Crea una rama feature (git checkout -b feature/AmazingFeature)

Commit tus cambios (git commit -m 'Add AmazingFeature')

Push a la rama (git push origin feature/AmazingFeature)

Abre un Pull Request

📝 Roadmap
Integración con bases de datos

Modelos de Machine Learning

Exportación a Excel/PDF

Dashboard en tiempo real

Múltiples usuarios

📄 Licencia
Distribuido bajo licencia MIT. Ver LICENSE para más información.

👨‍💻 Autor
NextiaSensei - GitHub - Nextia Marketing

🙏 Agradecimientos
Google Gemini AI

Flask Framework

Chart.js

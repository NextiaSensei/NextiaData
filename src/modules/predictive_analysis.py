import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def forecast_sales(df, date_column, value_column, periods=30):
    """
    Forecast inteligente de series temporales
    """
    try:
        # Preparar datos
        df_temp = df.copy()
        df_temp[date_column] = pd.to_datetime(df_temp[date_column])
        df_temp = df_temp.sort_values(date_column)
        
        # Método simple pero efectivo: promedio móvil + tendencia
        if len(df_temp) >= 7:
            # Calcular tendencia
            recent_data = df_temp[value_column].tail(7)
            x = np.arange(len(recent_data))
            slope = np.polyfit(x, recent_data.values, 1)[0]
            
            # Forecast basado en promedio móvil + tendencia
            base_value = recent_data.mean()
            forecast_values = [base_value + (slope * i) for i in range(1, periods + 1)]
            
            # Crear fechas de forecast
            last_date = df_temp[date_column].iloc[-1]
            forecast_dates = [last_date + timedelta(days=i) for i in range(1, periods + 1)]
            
            forecast_df = pd.DataFrame({
                'date': forecast_dates,
                'predicted': forecast_values,
                'confidence_lower': [max(0, v * 0.85) for v in forecast_values],
                'confidence_upper': [v * 1.15 for v in forecast_values]
            })
            
            return forecast_df.to_dict('records')
        
        return None
        
    except Exception as e:
        print(f"❌ Error en forecast: {e}")
        return None

def detect_anomalies(df, value_column, threshold=2.0):
    """
    Detectar valores anómalos usando Z-score
    """
    try:
        values = df[value_column].dropna()
        mean_val = values.mean()
        std_val = values.std()
        
        if std_val == 0:
            return []
            
        anomalies = []
        for idx, row in df.iterrows():
            z_score = abs((row[value_column] - mean_val) / std_val)
            if z_score > threshold:
                anomalies.append({
                    'index': idx,
                    'value': row[value_column],
                    'z_score': round(z_score, 2),
                    'message': f'Valor anómalo detectado: {row[value_column]} (Z-score: {z_score:.2f})'
                })
        
        return anomalies
        
    except Exception as e:
        print(f"❌ Error detectando anomalías: {e}")
        return []

def analyze_trends(df, date_column, value_column):
    """
    Análisis de tendencias y patrones
    """
    try:
        df_temp = df.copy()
        df_temp[date_column] = pd.to_datetime(df_temp[date_column])
        df_temp = df_temp.sort_values(date_column)
        
        if len(df_temp) < 2:
            return {"error": "Se necesitan al menos 2 puntos de datos"}
        
        # Calcular tendencia general
        x = np.arange(len(df_temp))
        y = df_temp[value_column].values
        slope, intercept = np.polyfit(x, y, 1)
        
        # Determinar dirección de la tendencia
        if slope > 0:
            trend_direction = "📈 CRECIENTE"
        elif slope < 0:
            trend_direction = "📉 DECRECIENTE"
        else:
            trend_direction = "➡️ ESTABLE"
        
        # Calcular métricas adicionales
        total_growth = ((y[-1] - y[0]) / y[0] * 100) if y[0] != 0 else 0
        avg_growth = total_growth / len(df_temp) if len(df_temp) > 0 else 0
        
        return {
            'trend_direction': trend_direction,
            'slope': round(slope, 4),
            'total_growth_percent': round(total_growth, 2),
            'average_daily_growth': round(avg_growth, 2),
            'current_value': y[-1],
            'starting_value': y[0],
            'data_points': len(df_temp)
        }
        
    except Exception as e:
        return {"error": f"Error analizando tendencias: {str(e)}"}

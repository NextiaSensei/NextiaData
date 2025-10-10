import pandas as pd
import numpy as np

def clean_dataset(df):
    """
    Limpieza automática de dataset
    """
    try:
        df_clean = df.copy()
        
        # Eliminar duplicados
        df_clean = df_clean.drop_duplicates()
        
        # Manejar valores nulos
        for col in df_clean.columns:
            if df_clean[col].dtype in ['float64', 'int64']:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
            else:
                df_clean[col].fillna('DESCONOCIDO', inplace=True)
        
        # Convertir columnas de fecha
        date_columns = []
        for col in df_clean.columns:
            if 'date' in col.lower() or 'fecha' in col.lower():
                try:
                    df_clean[col] = pd.to_datetime(df_clean[col])
                    date_columns.append(col)
                except:
                    pass
        
        return {
            'cleaned_data': df_clean,
            'cleaning_report': {
                'original_rows': len(df),
                'cleaned_rows': len(df_clean),
                'duplicates_removed': len(df) - len(df_clean),
                'date_columns_found': date_columns
            }
        }
        
    except Exception as e:
        return {"error": f"Error limpiando dataset: {str(e)}"}

def get_dataset_insights(df):
    """
    Generar insights automáticos del dataset
    """
    try:
        insights = []
        
        # Insights básicos
        insights.append(f"📊 Dataset con {len(df)} filas y {len(df.columns)} columnas")
        
        # Columnas numéricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            insights.append(f"🔢 Columnas numéricas: {', '.join(numeric_cols)}")
            
            # Estadísticas de columnas numéricas
            for col in numeric_cols[:3]:  # Primeras 3 columnas
                insights.append(f"   📈 {col}: Max {df[col].max():.2f}, Min {df[col].min():.2f}, Avg {df[col].mean():.2f}")
        
        # Columnas categóricas
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            insights.append(f"🏷️ Columnas categóricas: {', '.join(categorical_cols)}")
            
            for col in categorical_cols[:2]:  # Primeras 2 columnas
                top_categories = df[col].value_counts().head(3)
                insights.append(f"   📊 {col}: Top categorías - {', '.join([f'{k}({v})' for k, v in top_categories.items()])}")
        
        # Valores nulos
        null_counts = df.isnull().sum()
        total_nulls = null_counts.sum()
        if total_nulls > 0:
            insights.append(f"⚠️ Valores nulos encontrados: {total_nulls} total")
        
        return insights
        
    except Exception as e:
        return [f"❌ Error generando insights: {str(e)}"]

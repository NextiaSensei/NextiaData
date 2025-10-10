import pandas as pd
import numpy as np

def calculate_roi(campaign_cost, revenue_generated):
    """
    Calcular ROI de campañas de marketing
    """
    try:
        if campaign_cost <= 0:
            return {"error": "El costo de campaña debe ser mayor a 0"}
        
        roi = ((revenue_generated - campaign_cost) / campaign_cost) * 100
        profit = revenue_generated - campaign_cost
        
        return {
            'roi_percent': round(roi, 2),
            'profit': round(profit, 2),
            'campaign_cost': round(campaign_cost, 2),
            'revenue_generated': round(revenue_generated, 2),
            'break_even_point': campaign_cost
        }
        
    except Exception as e:
        return {"error": f"Error calculando ROI: {str(e)}"}

def calculate_cac(campaign_cost, customers_acquired):
    """
    Calcular Customer Acquisition Cost
    """
    try:
        if customers_acquired <= 0:
            return {"error": "Se necesitan clientes adquiridos mayores a 0"}
        
        cac = campaign_cost / customers_acquired
        
        return {
            'cac': round(cac, 2),
            'total_campaign_cost': round(campaign_cost, 2),
            'customers_acquired': customers_acquired,
            'efficiency': 'ALTA' if cac < 100 else 'MEDIA' if cac < 500 else 'BAJA'
        }
        
    except Exception as e:
        return {"error": f"Error calculando CAC: {str(e)}"}

def analyze_campaign_performance(df, cost_column, revenue_column, campaign_column=None):
    """
    Análisis de performance de campañas de marketing
    """
    try:
        results = {}
        
        # ROI general
        total_cost = df[cost_column].sum()
        total_revenue = df[revenue_column].sum()
        results['overall'] = calculate_roi(total_cost, total_revenue)
        
        # Análisis por campaña si existe la columna
        if campaign_column and campaign_column in df.columns:
            campaign_results = {}
            for campaign in df[campaign_column].unique():
                campaign_data = df[df[campaign_column] == campaign]
                camp_cost = campaign_data[cost_column].sum()
                camp_revenue = campaign_data[revenue_column].sum()
                campaign_results[campaign] = calculate_roi(camp_cost, camp_revenue)
            
            results['by_campaign'] = campaign_results
        
        # Métricas adicionales
        results['metrics'] = {
            'total_campaigns': len(df[campaign_column].unique()) if campaign_column else 1,
            'avg_roi': results['overall'].get('roi_percent', 0),
            'total_investment': total_cost,
            'total_return': total_revenue
        }
        
        return results
        
    except Exception as e:
        return {"error": f"Error analizando campañas: {str(e)}"}

def optimize_budget_allocation(historical_data, total_budget):
    """
    Optimizar asignación de presupuesto basado en performance histórico
    """
    try:
        # Agrupar por canal/campaña y calcular ROI
        performance = historical_data.groupby('channel').agg({
            'cost': 'sum',
            'revenue': 'sum',
            'conversions': 'sum'
        }).reset_index()
        
        performance['roi'] = ((performance['revenue'] - performance['cost']) / performance['cost']) * 100
        performance['efficiency'] = performance['conversions'] / performance['cost']
        
        # Normalizar scores para asignación
        performance['score'] = (performance['roi'] * 0.6 + performance['efficiency'] * 0.4)
        total_score = performance['score'].sum()
        
        if total_score > 0:
            performance['budget_allocation'] = (performance['score'] / total_score) * total_budget
        else:
            # Distribución equitativa si no hay datos históricos
            performance['budget_allocation'] = total_budget / len(performance)
        
        return performance.to_dict('records')
        
    except Exception as e:
        return {"error": f"Error optimizando presupuesto: {str(e)}"}

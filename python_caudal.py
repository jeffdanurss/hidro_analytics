import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)

def load_and_prepare_data():
    """Cargar y preparar los datos"""
    try:
        # Cargar datos
        precip_papallacta = pd.read_csv('precipitacion_papallacta.csv')
        precip_quijos = pd.read_csv('precipitacion_quijos.csv')
        caudal_papallacta = pd.read_csv('caudal_papallacta.csv')
        caudal_quijos = pd.read_csv('caudal_quijos.csv')
        
        print("✅ Datos cargados exitosamente")
        return precip_papallacta, precip_quijos, caudal_papallacta, caudal_quijos
    except Exception as e:
        print(f"❌ Error cargando datos: {e}")
        return None, None, None, None

def analyze_correlation_station(precip_data, caudal_data, station_name):
    """Analizar correlación para una estación específica"""
    print(f"\n{'='*60}")
    print(f"📊 ANÁLISIS PRECIPITACIÓN-CAUDAL: {station_name.upper()}")
    print(f"{'='*60}")
    
    # Crear DataFrames para análisis
    precip_df = pd.DataFrame()
    caudal_df = pd.DataFrame()
    
    # Procesar datos de precipitación
    precip_cols = [col for col in precip_data.columns if col != 'Fecha']
    caudal_cols = [col for col in caudal_data.columns if col != 'Fecha']
    
    # Convertir fechas si existen
    if 'Fecha' in precip_data.columns:
        precip_data['Fecha'] = pd.to_datetime(precip_data['Fecha'])
    if 'Fecha' in caudal_data.columns:
        caudal_data['Fecha'] = pd.to_datetime(caudal_data['Fecha'])
    
    # Calcular precipitación total por registro
    if precip_cols:
        precip_data['Precip_Total'] = precip_data[precip_cols].sum(axis=1)
    else:
        precip_data['Precip_Total'] = precip_data.iloc[:, 1:].sum(axis=1) if len(precip_data.columns) > 1 else precip_data.iloc[:, 0]
    
    # Calcular caudal promedio por registro
    if caudal_cols:
        caudal_data['Caudal_Prom'] = caudal_data[caudal_cols].mean(axis=1)
    else:
        caudal_data['Caudal_Prom'] = caudal_data.iloc[:, 1:].mean(axis=1) if len(caudal_data.columns) > 1 else caudal_data.iloc[:, 0]
    
    # Alinear datos por fecha o índice
    if 'Fecha' in precip_data.columns and 'Fecha' in caudal_data.columns:
        # Merge por fecha
        merged_data = pd.merge(precip_data[['Fecha', 'Precip_Total']], 
                              caudal_data[['Fecha', 'Caudal_Prom']], 
                              on='Fecha', how='inner')
    else:
        # Alinear por índice común (mínimo registros)
        min_len = min(len(precip_data), len(caudal_data))
        merged_data = pd.DataFrame({
            'Precip_Total': precip_data['Precip_Total'].iloc[:min_len],
            'Caudal_Prom': caudal_data['Caudal_Prom'].iloc[:min_len]
        })
    
    # Eliminar valores NaN
    merged_data = merged_data.dropna()
    
    if len(merged_data) < 10:
        print("⚠️  Insuficientes datos para análisis significativo")
        return None
    
    # Cálculos estadísticos
    precip = merged_data['Precip_Total']
    caudal = merged_data['Caudal_Prom']
    
    # Correlación de Pearson
    correlation, p_value = stats.pearsonr(precip, caudal)
    
    # Correlación de Spearman (no paramétrica)
    spearman_corr, spearman_p = stats.spearmanr(precip, caudal)
    
    print(f"📈 REGISTROS ANALIZADOS: {len(merged_data)}")
    print(f"🔗 CORRELACIÓN PEARSON: {correlation:.3f} (p-value: {p_value:.4f})")
    print(f"🔗 CORRELACIÓN SPEARMAN: {spearman_corr:.3f} (p-value: {spearman_p:.4f})")
    
    # Interpretación
    if p_value < 0.05:
        print("✅ CORRELACIÓN ESTADÍSTICAMENTE SIGNIFICATIVA")
        if correlation > 0.7:
            print("🎯 RELACIÓN MUY FUERTE POSITIVA")
        elif correlation > 0.5:
            print("📊 RELACIÓN MODERADA POSITIVA")
        elif correlation > 0.3:
            print("📈 RELACIÓN DÉBIL POSITIVA")
        elif correlation < -0.7:
            print("📉 RELACIÓN MUY FUERTE NEGATIVA")
        elif correlation < -0.5:
            print("📊 RELACIÓN MODERADA NEGATIVA")
        else:
            print("🔄 RELACIÓN MUY DÉBIL O INEXISTENTE")
    else:
        print("❌ CORRELACIÓN NO SIGNIFICATIVA ESTADÍSTICAMENTE")
    
    # Regresión lineal
    X = precip.values.reshape(-1, 1)
    y = caudal.values
    model = LinearRegression().fit(X, y)
    r2 = r2_score(y, model.predict(X))
    
    print(f"📊 COEFICIENTE DE DETERMINACIÓN (R²): {r2:.3f}")
    print(f"📏 ECUACIÓN: Caudal = {model.coef_[0]:.4f} × Precip + {model.intercept_:.2f}")
    
    # Estadísticas descriptivas
    print(f"\n📋 ESTADÍSTICAS DESCRIPTIVAS:")
    print(f"   Precipitación - Media: {precip.mean():.2f}, Desv: {precip.std():.2f}")
    print(f"   Caudal - Media: {caudal.mean():.2f}, Desv: {caudal.std():.2f}")
    
    return merged_data, correlation, r2, model

def create_visualizations(data, station_name, model):
    """Crear visualizaciones del análisis"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Análisis Precipitación-Caudal: {station_name}', fontsize=16, fontweight='bold')
    
    precip = data['Precip_Total']
    caudal = data['Caudal_Prom']
    
    # 1. Diagrama de dispersión
    axes[0, 0].scatter(precip, caudal, alpha=0.6, color='blue')
    axes[0, 0].set_xlabel('Precipitación Total')
    axes[0, 0].set_ylabel('Caudal Promedio')
    axes[0, 0].set_title('Diagrama de Dispersión Precipitación-Caudal')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Línea de regresión
    precip_sorted = np.sort(precip)
    caudal_pred = model.predict(precip_sorted.reshape(-1, 1))
    axes[0, 0].plot(precip_sorted, caudal_pred, 'r-', linewidth=2, 
                    label=f'R² = {r2_score(data["Caudal_Prom"], model.predict(data["Precip_Total"].values.reshape(-1, 1))):.3f}')
    axes[0, 0].legend()
    
    # 2. Histogramas
    axes[0, 1].hist(precip, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 1].set_xlabel('Precipitación Total')
    axes[0, 1].set_ylabel('Frecuencia')
    axes[0, 1].set_title('Distribución de Precipitación')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].hist(caudal, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1, 0].set_xlabel('Caudal Promedio')
    axes[1, 0].set_ylabel('Frecuencia')
    axes[1, 0].set_title('Distribución de Caudal')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 3. Serie temporal (si hay fechas)
    if len(data) > 1:
        axes[1, 1].plot(range(len(precip)), precip, 'b-', alpha=0.7, label='Precipitación')
        ax2 = axes[1, 1].twinx()
        ax2.plot(range(len(caudal)), caudal, 'g-', alpha=0.7, label='Caudal')
        axes[1, 1].set_xlabel('Índice Temporal')
        axes[1, 1].set_ylabel('Precipitación', color='b')
        ax2.set_ylabel('Caudal', color='g')
        axes[1, 1].set_title('Series Temporales')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'analisis_{station_name.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()

def main_correlation_analysis():
    """Función principal para análisis de correlación"""
    print("🌊 ANÁLISIS DE CORRELACIÓN PRECIPITACIÓN-CAUDAL")
    print("="*60)
    
    # Cargar datos
    precip_papallacta, precip_quijos, caudal_papallacta, caudal_quijos = load_and_prepare_data()
    
    if precip_papallacta is None:
        return
    
    results = {}
    
    # Análisis Papallacta
    result_papallacta = analyze_correlation_station(
        precip_papallacta, caudal_papallacta, "Papallacta"
    )
    if result_papallacta:
        data_p, corr_p, r2_p, model_p = result_papallacta
        create_visualizations(data_p, "Papallacta", model_p)
        results['Papallacta'] = {'correlation': corr_p, 'r2': r2_p}
    
    # Análisis Quijos
    result_quijos = analyze_correlation_station(
        precip_quijos, caudal_quijos, "Quijos"
    )
    if result_quijos:
        data_q, corr_q, r2_q, model_q = result_quijos
        create_visualizations(data_q, "Quijos", model_q)
        results['Quijos'] = {'correlation': corr_q, 'r2': r2_q}
    
    # Resumen final
    print(f"\n{'='*60}")
    print("📋 RESUMEN GENERAL")
    print(f"{'='*60}")
    
    for station, metrics in results.items():
        print(f"\n📍 {station}:")
        print(f"   Correlación: {metrics['correlation']:.3f}")
        print(f"   R²: {metrics['r2']:.3f}")
        
        if abs(metrics['correlation']) > 0.7:
            print("   🎯 Relación fuerte")
        elif abs(metrics['correlation']) > 0.5:
            print("   📊 Relación moderada")
        elif abs(metrics['correlation']) > 0.3:
            print("   📈 Relación débil")
        else:
            print("   🔄 Relación muy débil")
    
    return results

# Ejecutar análisis
if __name__ == "__main__":
    results = main_correlation_analysis()
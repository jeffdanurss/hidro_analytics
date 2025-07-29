import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)

def load_level_flow_data():
    """Cargar datos de caudal y nivel"""
    try:
        caudal_papallacta = pd.read_csv('caudal_papallacta.csv')
        nivel_papallacta = pd.read_csv('nivel_papallacta.csv')
        caudal_quijos = pd.read_csv('caudal_quijos.csv')
        nivel_quijos = pd.read_csv('nivel_quijos.csv')
        
        print("✅ Datos cargados exitosamente")
        return caudal_papallacta, nivel_papallacta, caudal_quijos, nivel_quijos
    except Exception as e:
        print(f"❌ Error cargando datos: {e}")
        return None, None, None, None

def prepare_station_data(caudal_data, nivel_data, station_name):
    """Preparar datos para una estación"""
    print(f"\n📍 PROCESANDO DATOS: {station_name}")
    
    # Procesar caudal - promedio de todas las estaciones
    caudal_cols = [col for col in caudal_data.columns if col != 'Fecha']
    if caudal_cols:
        caudal_data['Caudal_Prom'] = caudal_data[caudal_cols].mean(axis=1)
    else:
        caudal_data['Caudal_Prom'] = caudal_data.iloc[:, 1:].mean(axis=1)
    
    # Procesar nivel - promedio de todas las estaciones
    nivel_cols = [col for col in nivel_data.columns if col != 'Fecha']
    if nivel_cols:
        nivel_data['Nivel_Prom'] = nivel_data[nivel_cols].mean(axis=1)
    else:
        nivel_data['Nivel_Prom'] = nivel_data.iloc[:, 1:].mean(axis=1)
    
    # Alinear datos por fecha
    if 'Fecha' in caudal_data.columns and 'Fecha' in nivel_data.columns:
        caudal_data['Fecha'] = pd.to_datetime(caudal_data['Fecha'])
        nivel_data['Fecha'] = pd.to_datetime(nivel_data['Fecha'])
        
        # Merge por fecha
        merged_data = pd.merge(caudal_data[['Fecha', 'Caudal_Prom']], 
                              nivel_data[['Fecha', 'Nivel_Prom']], 
                              on='Fecha', how='inner')
    else:
        # Alinear por índice
        min_len = min(len(caudal_data), len(nivel_data))
        merged_data = pd.DataFrame({
            'Caudal_Prom': caudal_data['Caudal_Prom'].iloc[:min_len],
            'Nivel_Prom': nivel_data['Nivel_Prom'].iloc[:min_len]
        })
    
    # Eliminar valores NaN
    merged_data = merged_data.dropna()
    
    print(f"📊 Registros alineados: {len(merged_data)}")
    return merged_data

def analyze_flow_level_relationship(data, station_name):
    """Analizar relación caudal-nivel"""
    print(f"\n{'='*60}")
    print(f"📊 ANÁLISIS CAUDAL-NIVEL: {station_name.upper()}")
    print(f"{'='*60}")
    
    if len(data) < 5:
        print("⚠️  Insuficientes datos para análisis")
        return None
    
    caudal = data['Caudal_Prom']
    nivel = data['Nivel_Prom']
    
    # Correlaciones
    pearson_corr, pearson_p = stats.pearsonr(caudal, nivel)
    spearman_corr, spearman_p = stats.spearmanr(caudal, nivel)
    
    print(f"📈 REGISTROS ANALIZADOS: {len(data)}")
    print(f"🔗 CORRELACIÓN PEARSON: {pearson_corr:.3f} (p-value: {pearson_p:.4f})")
    print(f"🔗 CORRELACIÓN SPEARMAN: {spearman_corr:.3f} (p-value: {spearman_p:.4f})")
    
    # Interpretación
    if pearson_p < 0.05:
        print("✅ CORRELACIÓN ESTADÍSTICAMENTE SIGNIFICATIVA")
        if abs(pearson_corr) > 0.8:
            print("🎯 RELACIÓN MUY FUERTE")
        elif abs(pearson_corr) > 0.6:
            print("📊 RELACIÓN MODERADA-FUERTE")
        elif abs(pearson_corr) > 0.4:
            print("📈 RELACIÓN MODERADA")
        else:
            print("🔄 RELACIÓN DÉBIL")
    else:
        print("❌ CORRELACIÓN NO SIGNIFICATIVA ESTADÍSTICAMENTE")
    
    # Modelos de regresión
    results = {}
    
    # 1. Regresión lineal
    X_linear = nivel.values.reshape(-1, 1)
    y_linear = caudal.values
    linear_model = LinearRegression().fit(X_linear, y_linear)
    linear_r2 = r2_score(y_linear, linear_model.predict(X_linear))
    linear_rmse = np.sqrt(mean_squared_error(y_linear, linear_model.predict(X_linear)))
    
    results['linear'] = {
        'model': linear_model,
        'r2': linear_r2,
        'rmse': linear_rmse,
        'equation': f"Caudal = {linear_model.coef_[0]:.4f} × Nivel + {linear_model.intercept_:.2f}"
    }
    
    # 2. Regresión polinómica (cuadrática)
    poly_features = PolynomialFeatures(degree=2)
    X_poly = poly_features.fit_transform(nivel.values.reshape(-1, 1))
    poly_model = LinearRegression().fit(X_poly, y_linear)
    poly_r2 = r2_score(y_linear, poly_model.predict(X_poly))
    poly_rmse = np.sqrt(mean_squared_error(y_linear, poly_model.predict(X_poly)))
    
    results['polynomial'] = {
        'model': poly_model,
        'poly_features': poly_features,
        'r2': poly_r2,
        'rmse': poly_rmse
    }
    
    # Mostrar resultados
    print(f"\n📊 MODELOS DE REGRESIÓN:")
    print(f"   Lineal - R²: {linear_r2:.3f}, RMSE: {linear_rmse:.2f}")
    print(f"   Ecuación: {results['linear']['equation']}")
    print(f"   Polinómica - R²: {poly_r2:.3f}, RMSE: {poly_rmse:.2f}")
    
    # Estadísticas descriptivas
    print(f"\n📋 ESTADÍSTICAS DESCRIPTIVAS:")
    print(f"   Caudal - Media: {caudal.mean():.2f}, Desv: {caudal.std():.2f}")
    print(f"   Nivel - Media: {nivel.mean():.2f}, Desv: {nivel.std():.2f}")
    print(f"   Rango Caudal: {caudal.min():.2f} - {caudal.max():.2f}")
    print(f"   Rango Nivel: {nivel.min():.2f} - {nivel.max():.2f}")
    
    return results

def create_flow_level_visualizations(data, station_name, results):
    """Crear visualizaciones del análisis caudal-nivel"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Análisis Caudal-Nivel: {station_name}', fontsize=16, fontweight='bold')
    
    caudal = data['Caudal_Prom']
    nivel = data['Nivel_Prom']
    
    # 1. Diagrama de dispersión con regresión lineal
    axes[0, 0].scatter(nivel, caudal, alpha=0.6, color='blue', s=50)
    axes[0, 0].set_xlabel('Nivel de Agua')
    axes[0, 0].set_ylabel('Caudal')
    axes[0, 0].set_title('Relación Caudal-Nivel (Lineal)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Línea de regresión lineal
    nivel_sorted = np.sort(nivel)
    linear_pred = results['linear']['model'].predict(nivel_sorted.reshape(-1, 1))
    axes[0, 0].plot(nivel_sorted, linear_pred, 'r-', linewidth=2, 
                    label=f'R² = {results["linear"]["r2"]:.3f}')
    axes[0, 0].legend()
    
    # 2. Diagrama de dispersión con regresión polinómica
    axes[0, 1].scatter(nivel, caudal, alpha=0.6, color='green', s=50)
    axes[0, 1].set_xlabel('Nivel de Agua')
    axes[0, 1].set_ylabel('Caudal')
    axes[0, 1].set_title('Relación Caudal-Nivel (Polinómica)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Línea de regresión polinómica
    nivel_sorted = np.sort(nivel)
    X_poly_sorted = results['polynomial']['poly_features'].transform(nivel_sorted.reshape(-1, 1))
    poly_pred = results['polynomial']['model'].predict(X_poly_sorted)
    axes[0, 1].plot(nivel_sorted, poly_pred, 'orange', linewidth=2,
                    label=f'R² = {results["polynomial"]["r2"]:.3f}')
    axes[0, 1].legend()
    
    # 3. Histogramas
    axes[1, 0].hist(caudal, bins=20, alpha=0.7, color='lightblue', edgecolor='black')
    axes[1, 0].set_xlabel('Caudal (m³/s)')
    axes[1, 0].set_ylabel('Frecuencia')
    axes[1, 0].set_title('Distribución de Caudal')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].hist(nivel, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1, 1].set_xlabel('Nivel de Agua')
    axes[1, 1].set_ylabel('Frecuencia')
    axes[1, 1].set_title('Distribución de Nivel')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'caudal_nivel_{station_name.lower().replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def create_rating_curve(data, station_name, results):
    """Crear curva de rating (relación caudal-nivel)"""
    plt.figure(figsize=(10, 8))
    
    caudal = data['Caudal_Prom']
    nivel = data['Nivel_Prom']
    
    # Scatter plot
    plt.scatter(nivel, caudal, alpha=0.6, color='blue', s=60, label='Datos observados')
    
    # Curva de rating (modelo polinómico)
    nivel_sorted = np.sort(nivel)
    X_poly_sorted = results['polynomial']['poly_features'].transform(nivel_sorted.reshape(-1, 1))
    poly_pred = results['polynomial']['model'].predict(X_poly_sorted)
    
    plt.plot(nivel_sorted, poly_pred, 'red', linewidth=2, 
             label=f'Curva de Rating (R² = {results["polynomial"]["r2"]:.3f})')
    
    plt.xlabel('Nivel de Agua')
    plt.ylabel('Caudal (m³/s)')
    plt.title(f'Curva de Rating - {station_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Agregar ecuación
    plt.text(0.05, 0.95, f'R² = {results["polynomial"]["r2"]:.3f}', 
             transform=plt.gca().transAxes, fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'curva_rating_{station_name.lower().replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def main_flow_level_analysis():
    """Función principal para análisis caudal-nivel"""
    print("🌊 ANÁLISIS DE RELACIÓN CAUDAL-NIVEL")
    print("="*60)
    
    # Cargar datos
    caudal_papallacta, nivel_papallacta, caudal_quijos, nivel_quijos = load_level_flow_data()
    
    if caudal_papallacta is None:
        return
    
    all_results = {}
    
    # Análisis Papallacta
    print("\n" + "="*80)
    data_papallacta = prepare_station_data(caudal_papallacta, nivel_papallacta, "Papallacta")
    if len(data_papallacta) > 0:
        results_papallacta = analyze_flow_level_relationship(data_papallacta, "Papallacta")
        if results_papallacta:
            create_flow_level_visualizations(data_papallacta, "Papallacta", results_papallacta)
            create_rating_curve(data_papallacta, "Papallacta", results_papallacta)
            all_results['Papallacta'] = results_papallacta
    
    # Análisis Quijos
    print("\n" + "="*80)
    data_quijos = prepare_station_data(caudal_quijos, nivel_quijos, "Quijos")
    if len(data_quijos) > 0:
        results_quijos = analyze_flow_level_relationship(data_quijos, "Quijos")
        if results_quijos:
            create_flow_level_visualizations(data_quijos, "Quijos", results_quijos)
            create_rating_curve(data_quijos, "Quijos", results_quijos)
            all_results['Quijos'] = results_quijos
    
    # Resumen comparativo
    print(f"\n{'='*80}")
    print("📋 RESUMEN COMPARATIVO")
    print(f"{'='*80}")
    
    for station, results in all_results.items():
        if results:
            print(f"\n📍 {station}:")
            print(f"   Correlación Lineal R²: {results['linear']['r2']:.3f}")
            print(f"   Correlación Polinómica R²: {results['polynomial']['r2']:.3f}")
            print(f"   Error RMSE Lineal: {results['linear']['rmse']:.2f}")
            print(f"   Error RMSE Polinómico: {results['polynomial']['rmse']:.2f}")
            
            if results['polynomial']['r2'] > results['linear']['r2']:
                print("   🎯 Modelo polinómico más adecuado")
            else:
                print("   📊 Modelo lineal más adecuado")
    
    return all_results

# Ejecutar análisis
if __name__ == "__main__":
    results = main_flow_level_analysis()
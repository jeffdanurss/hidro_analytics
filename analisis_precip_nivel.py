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

def load_precip_level_data():
    """Cargar datos de precipitación y nivel"""
    try:
        precip_papallacta = pd.read_csv('precipitacion_papallacta.csv')
        nivel_papallacta = pd.read_csv('nivel_papallacta.csv')
        precip_quijos = pd.read_csv('precipitacion_quijos.csv')
        nivel_quijos = pd.read_csv('nivel_quijos.csv')
        
        print("✅ Datos cargados exitosamente")
        return precip_papallacta, nivel_papallacta, precip_quijos, nivel_quijos
    except Exception as e:
        print(f"❌ Error cargando datos: {e}")
        return None, None, None, None

def prepare_precip_level_data(precip_data, nivel_data, station_name):
    """Preparar datos de precipitación y nivel para una estación"""
    print(f"\n📍 PROCESANDO DATOS: {station_name}")
    
    # Procesar precipitación - suma total por registro
    precip_cols = [col for col in precip_data.columns if col != 'Fecha']
    if precip_cols:
        precip_data['Precip_Total'] = precip_data[precip_cols].sum(axis=1)
    else:
        precip_data['Precip_Total'] = precip_data.iloc[:, 1:].sum(axis=1)
    
    # Procesar nivel - promedio de todas las estaciones
    nivel_cols = [col for col in nivel_data.columns if col != 'Fecha']
    if nivel_cols:
        nivel_data['Nivel_Prom'] = nivel_data[nivel_cols].mean(axis=1)
    else:
        nivel_data['Nivel_Prom'] = nivel_data.iloc[:, 1:].mean(axis=1)
    
    # Alinear datos por fecha
    if 'Fecha' in precip_data.columns and 'Fecha' in nivel_data.columns:
        precip_data['Fecha'] = pd.to_datetime(precip_data['Fecha'])
        nivel_data['Fecha'] = pd.to_datetime(nivel_data['Fecha'])
        
        # Merge por fecha
        merged_data = pd.merge(precip_data[['Fecha', 'Precip_Total']], 
                              nivel_data[['Fecha', 'Nivel_Prom']], 
                              on='Fecha', how='inner')
    else:
        # Alinear por índice
        min_len = min(len(precip_data), len(nivel_data))
        merged_data = pd.DataFrame({
            'Precip_Total': precip_data['Precip_Total'].iloc[:min_len],
            'Nivel_Prom': nivel_data['Nivel_Prom'].iloc[:min_len]
        })
    
    # Eliminar valores NaN
    merged_data = merged_data.dropna()
    
    print(f"📊 Registros alineados: {len(merged_data)}")
    return merged_data

def analyze_precip_level_relationship(data, station_name):
    """Analizar relación precipitación-nivel"""
    print(f"\n{'='*60}")
    print(f"📊 ANÁLISIS PRECIPITACIÓN-NIVEL: {station_name.upper()}")
    print(f"{'='*60}")
    
    if len(data) < 5:
        print("⚠️  Insuficientes datos para análisis")
        return None
    
    precip = data['Precip_Total']
    nivel = data['Nivel_Prom']
    
    # Correlaciones
    pearson_corr, pearson_p = stats.pearsonr(precip, nivel)
    spearman_corr, spearman_p = stats.spearmanr(precip, nivel)
    
    print(f"📈 REGISTROS ANALIZADOS: {len(data)}")
    print(f"🔗 CORRELACIÓN PEARSON: {pearson_corr:.3f} (p-value: {pearson_p:.4f})")
    print(f"🔗 CORRELACIÓN SPEARMAN: {spearman_corr:.3f} (p-value: {spearman_p:.4f})")
    
    # Interpretación
    if pearson_p < 0.05:
        print("✅ CORRELACIÓN ESTADÍSTICAMENTE SIGNIFICATIVA")
        if abs(pearson_corr) > 0.7:
            print("🎯 RELACIÓN MUY FUERTE")
        elif abs(pearson_corr) > 0.5:
            print("📊 RELACIÓN MODERADA-FUERTE")
        elif abs(pearson_corr) > 0.3:
            print("📈 RELACIÓN MODERADA")
        else:
            print("🔄 RELACIÓN DÉBIL")
    else:
        print("❌ CORRELACIÓN NO SIGNIFICATIVA ESTADÍSTICAMENTE")
    
    # Análisis por cuartiles
    print(f"\n📊 ANÁLISIS POR CUARTILES:")
    precip_q1 = precip.quantile(0.25)
    precip_q2 = precip.quantile(0.50)
    precip_q3 = precip.quantile(0.75)
    
    nivel_q1 = nivel.quantile(0.25)
    nivel_q2 = nivel.quantile(0.50)
    nivel_q3 = nivel.quantile(0.75)
    
    print(f"   Precipitación - Q1: {precip_q1:.2f}, Q2: {precip_q2:.2f}, Q3: {precip_q3:.2f}")
    print(f"   Nivel - Q1: {nivel_q1:.2f}, Q2: {nivel_q2:.2f}, Q3: {nivel_q3:.2f}")
    
    # Modelos de regresión
    results = {}
    
    # 1. Regresión lineal
    X_linear = precip.values.reshape(-1, 1)
    y_linear = nivel.values
    linear_model = LinearRegression().fit(X_linear, y_linear)
    linear_r2 = r2_score(y_linear, linear_model.predict(X_linear))
    linear_rmse = np.sqrt(mean_squared_error(y_linear, linear_model.predict(X_linear)))
    
    results['linear'] = {
        'model': linear_model,
        'r2': linear_r2,
        'rmse': linear_rmse,
        'equation': f"Nivel = {linear_model.coef_[0]:.4f} × Precip + {linear_model.intercept_:.2f}"
    }
    
    # 2. Regresión polinómica (cuadrática)
    poly_features = PolynomialFeatures(degree=2)
    X_poly = poly_features.fit_transform(precip.values.reshape(-1, 1))
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
    print(f"   Precipitación - Media: {precip.mean():.2f}, Desv: {precip.std():.2f}")
    print(f"   Nivel - Media: {nivel.mean():.2f}, Desv: {nivel.std():.2f}")
    print(f"   Rango Precip: {precip.min():.2f} - {precip.max():.2f}")
    print(f"   Rango Nivel: {nivel.min():.2f} - {nivel.max():.2f}")
    
    return results

def create_precip_level_visualizations(data, station_name, results):
    """Crear visualizaciones del análisis precipitación-nivel"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Análisis Precipitación-Nivel: {station_name}', fontsize=16, fontweight='bold')
    
    precip = data['Precip_Total']
    nivel = data['Nivel_Prom']
    
    # 1. Diagrama de dispersión con regresión lineal
    axes[0, 0].scatter(precip, nivel, alpha=0.6, color='blue', s=50)
    axes[0, 0].set_xlabel('Precipitación Total')
    axes[0, 0].set_ylabel('Nivel de Agua')
    axes[0, 0].set_title('Relación Precipitación-Nivel (Lineal)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Línea de regresión lineal
    precip_sorted = np.sort(precip)
    linear_pred = results['linear']['model'].predict(precip_sorted.reshape(-1, 1))
    axes[0, 0].plot(precip_sorted, linear_pred, 'r-', linewidth=2, 
                    label=f'R² = {results["linear"]["r2"]:.3f}')
    axes[0, 0].legend()
    
    # 2. Diagrama de dispersión con regresión polinómica
    axes[0, 1].scatter(precip, nivel, alpha=0.6, color='green', s=50)
    axes[0, 1].set_xlabel('Precipitación Total')
    axes[0, 1].set_ylabel('Nivel de Agua')
    axes[0, 1].set_title('Relación Precipitación-Nivel (Polinómica)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Línea de regresión polinómica
    precip_sorted = np.sort(precip)
    X_poly_sorted = results['polynomial']['poly_features'].transform(precip_sorted.reshape(-1, 1))
    poly_pred = results['polynomial']['model'].predict(X_poly_sorted)
    axes[0, 1].plot(precip_sorted, poly_pred, 'orange', linewidth=2,
                    label=f'R² = {results["polynomial"]["r2"]:.3f}')
    axes[0, 1].legend()
    
    # 3. Histogramas
    axes[1, 0].hist(precip, bins=20, alpha=0.7, color='lightblue', edgecolor='black')
    axes[1, 0].set_xlabel('Precipitación Total')
    axes[1, 0].set_ylabel('Frecuencia')
    axes[1, 0].set_title('Distribución de Precipitación')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].hist(nivel, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1, 1].set_xlabel('Nivel de Agua')
    axes[1, 1].set_ylabel('Frecuencia')
    axes[1, 1].set_title('Distribución de Nivel')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'precip_nivel_{station_name.lower().replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def analyze_lag_correlation(precip_data, nivel_data, station_name, max_lag=10):
    """Analizar correlación con diferentes rezagos temporales"""
    print(f"\n{'='*60}")
    print(f"📊 ANÁLISIS DE REZAGOS TEMPORALES: {station_name.upper()}")
    print(f"{'='*60}")
    
    # Preparar datos
    precip = precip_data['Precip_Total']
    nivel = nivel_data['Nivel_Prom']
    
    min_len = min(len(precip), len(nivel))
    precip = precip.iloc[:min_len]
    nivel = nivel.iloc[:min_len]
    
    # Calcular correlaciones para diferentes rezagos
    lags = range(0, min(max_lag + 1, len(precip)//2))
    correlations = []
    
    for lag in lags:
        if lag == 0:
            corr, _ = stats.pearsonr(precip, nivel)
        else:
            corr, _ = stats.pearsonr(precip.iloc[:-lag], nivel.iloc[lag:])
        correlations.append(corr)
    
    # Encontrar mejor rezago
    best_lag = lags[np.argmax(correlations)]
    best_corr = max(correlations)
    
    print(f"📈 MEJOR CORRELACIÓN ENCONTRADA:")
    print(f"   Rezago óptimo: {best_lag} período(s)")
    print(f"   Correlación máxima: {best_corr:.3f}")
    
    # Visualización de rezagos
    plt.figure(figsize=(10, 6))
    plt.plot(lags, correlations, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Rezago (períodos)')
    plt.ylabel('Correlación')
    plt.title(f'Correlación Precipitación-Nivel vs Rezago - {station_name}')
    plt.grid(True, alpha=0.3)
    plt.axvline(x=best_lag, color='red', linestyle='--', alpha=0.7, 
                label=f'Mejor rezago: {best_lag}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'rezagos_{station_name.lower().replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return best_lag, best_corr

def main_precip_level_analysis():
    """Función principal para análisis precipitación-nivel"""
    print("🌧️ ANÁLISIS DE RELACIÓN PRECIPITACIÓN-NIVEL")
    print("="*60)
    
    # Cargar datos
    precip_papallacta, nivel_papallacta, precip_quijos, nivel_quijos = load_precip_level_data()
    
    if precip_papallacta is None:
        return
    
    all_results = {}
    
    # Análisis Papallacta
    print("\n" + "="*80)
    data_papallacta = prepare_precip_level_data(precip_papallacta, nivel_papallacta, "Papallacta")
    if len(data_papallacta) > 0:
        results_papallacta = analyze_precip_level_relationship(data_papallacta, "Papallacta")
        if results_papallacta:
            create_precip_level_visualizations(data_papallacta, "Papallacta", results_papallacta)
            
            # Análisis de rezagos
            try:
                best_lag, best_corr = analyze_lag_correlation(
                    precip_papallacta[['Precip_Total']], 
                    nivel_papallacta[['Nivel_Prom']], 
                    "Papallacta"
                )
                results_papallacta['best_lag'] = best_lag
                results_papallacta['best_lag_corr'] = best_corr
            except:
                print("⚠️  No se pudo calcular análisis de rezagos para Papallacta")
            
            all_results['Papallacta'] = results_papallacta
    
    # Análisis Quijos
    print("\n" + "="*80)
    data_quijos = prepare_precip_level_data(precip_quijos, nivel_quijos, "Quijos")
    if len(data_quijos) > 0:
        results_quijos = analyze_precip_level_relationship(data_quijos, "Quijos")
        if results_quijos:
            create_precip_level_visualizations(data_quijos, "Quijos", results_quijos)
            
            # Análisis de rezagos
            try:
                best_lag, best_corr = analyze_lag_correlation(
                    precip_quijos[['Precip_Total']], 
                    nivel_quijos[['Nivel_Prom']], 
                    "Quijos"
                )
                results_quijos['best_lag'] = best_lag
                results_quijos['best_lag_corr'] = best_corr
            except:
                print("⚠️  No se pudo calcular análisis de rezagos para Quijos")
            
            all_results['Quijos'] = results_quijos
    
    # Resumen comparativo
    print(f"\n{'='*80}")
    print("📋 RESUMEN COMPARATIVO")
    print(f"{'='*80}")
    
    for station, results in all_results.items():
        if results:
            print(f"\n📍 {station}:")
            print(f"   Correlación directa: {results['linear']['r2']:.3f}")
            print(f"   Correlación polinómica: {results['polynomial']['r2']:.3f}")
            if 'best_lag' in results:
                print(f"   Mejor rezago: {results['best_lag']} períodos")
                print(f"   Correlación con rezago: {results['best_lag_corr']:.3f}")
            
            if results['polynomial']['r2'] > results['linear']['r2']:
                print("   🎯 Modelo polinómico más adecuado")
            else:
                print("   📊 Modelo lineal más adecuado")
    
    return all_results

# Ejecutar análisis
if __name__ == "__main__":
    results = main_precip_level_analysis()
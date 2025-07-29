import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
from scipy import interpolate
from sklearn.impute import KNNImputer
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def assess_data_quality(df, data_type):
    """Evalúa la calidad de los datos y reporta datos faltantes"""
    print(f"\n=== CALIDAD DE DATOS: {data_type.upper()} ===")
    
    # Columnas numéricas (excluyendo fecha)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    total_records = len(df)
    print(f"Total de registros: {total_records}")
    
    # Análisis por columna
    missing_summary = []
    for col in numeric_cols:
        missing_count = df[col].isnull().sum()
        missing_pct = (missing_count / total_records) * 100
        
        missing_summary.append({
            'columna': col,
            'faltantes': missing_count,
            'porcentaje': missing_pct,
            'disponibles': total_records - missing_count
        })
        
        print(f"  {col}: {missing_count} faltantes ({missing_pct:.1f}%)")
    
    # Evaluación general
    if missing_summary:
        total_missing_pct = np.mean([item['porcentaje'] for item in missing_summary])
    else:
        total_missing_pct = 0
    
    if total_missing_pct > 30:
        quality_status = "🔴 CRÍTICA"
        recommendation = "Requiere imputación o búsqueda de datos adicionales"
    elif total_missing_pct > 15:
        quality_status = "🟡 MODERADA"
        recommendation = "Aplicar técnicas de imputación"
    else:
        quality_status = "🟢 BUENA"
        recommendation = "Proceder con análisis estándar"
    
    print(f"\nCalidad general: {quality_status}")
    print(f"Recomendación: {recommendation}")
    
    return missing_summary, total_missing_pct

def handle_missing_data_optimized(df, method='hybrid', max_gap=7, station_threshold=0.5):
    """
    Maneja datos faltantes con estrategias optimizadas para datos hidrológicos
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame con datos faltantes
    method : str
        'hybrid', 'interpolation', 'seasonal', 'station_average'
    max_gap : int
        Máximo número de días consecutivos a interpolar
    station_threshold : float
        Mínimo porcentaje de estaciones requeridas para datos confiables
    """
    
    df_filled = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    print(f"\nAplicando método OPTIMIZADO: {method.upper()}")
    
    if method == 'hybrid':
        # Estrategia híbrida optimizada para datos hidrológicos
        
        for col in numeric_cols:
            original_missing = df_filled[col].isnull().sum()
            
            # Paso 1: Interpolación para gaps pequeños (≤3 días)
            df_filled[col] = df_filled[col].interpolate(
                method='linear', 
                limit=3,
                limit_direction='both'
            )
            
            # Paso 2: Imputación estacional para gaps medianos
            if 'Fecha' in df_filled.columns:
                df_filled['mes'] = df_filled['Fecha'].dt.month
                df_filled['año'] = df_filled['Fecha'].dt.year
                
                # Promedio mensual de años anteriores
                monthly_patterns = df_filled.groupby('mes')[col].mean()
                
                for mes in range(1, 13):
                    mask = (df_filled['mes'] == mes) & (df_filled[col].isnull())
                    if mes in monthly_patterns.index and not pd.isna(monthly_patterns[mes]):
                        df_filled.loc[mask, col] = monthly_patterns[mes]
                
                df_filled.drop(['mes', 'año'], axis=1, inplace=True, errors='ignore')
            
            # Paso 3: Imputación por estaciones vecinas (para datos hidrológicos)
            if len(numeric_cols) > 1:
                # Calcular correlaciones entre estaciones
                correlations = df_filled[numeric_cols].corr()
                
                for idx in df_filled[df_filled[col].isnull()].index:
                    # Buscar estaciones altamente correlacionadas
                    if col in correlations.columns:
                        corr_stations = correlations[col].abs().sort_values(ascending=False)
                        corr_stations = corr_stations.drop(col, errors='ignore')
                        
                        for corr_station in corr_stations.index[:3]:  # Top 3 correlacionadas
                            if not pd.isna(df_filled.loc[idx, corr_station]) and corr_stations[corr_station] > 0.5:
                                # Usar ratio de estaciones correlacionadas
                                ratio = df_filled[col].mean() / df_filled[corr_station].mean()
                                if not pd.isna(ratio) and ratio > 0:
                                    df_filled.loc[idx, col] = df_filled.loc[idx, corr_station] * ratio
                                    break
            
            filled_missing = df_filled[col].isnull().sum()
            filled_count = original_missing - filled_missing
            improvement = (filled_count / original_missing * 100) if original_missing > 0 else 0
            
            print(f"  {col}: {filled_count} valores imputados ({improvement:.1f}% mejora), {filled_missing} aún faltantes")
    
    elif method == 'seasonal':
        # Imputación estacional avanzada
        for col in numeric_cols:
            if 'Fecha' in df_filled.columns:
                df_filled['mes'] = df_filled['Fecha'].dt.month
                df_filled['trimestre'] = df_filled['Fecha'].dt.quarter
                
                # Primero intentar con promedios mensuales
                monthly_means = df_filled.groupby('mes')[col].mean()
                for mes in range(1, 13):
                    mask = (df_filled['mes'] == mes) & (df_filled[col].isnull())
                    if mes in monthly_means.index and not pd.isna(monthly_means[mes]):
                        df_filled.loc[mask, col] = monthly_means[mes]
                
                # Luego con promedios trimestrales para los restantes
                quarterly_means = df_filled.groupby('trimestre')[col].mean()
                for trimestre in range(1, 5):
                    mask = (df_filled['trimestre'] == trimestre) & (df_filled[col].isnull())
                    if trimestre in quarterly_means.index and not pd.isna(quarterly_means[trimestre]):
                        df_filled.loc[mask, col] = quarterly_means[trimestre]
                
                df_filled.drop(['mes', 'trimestre'], axis=1, inplace=True, errors='ignore')
    
    elif method == 'station_average':
        # Promedio de estaciones disponibles
        if len(numeric_cols) > 1:
            for idx in df_filled.index:
                for col in numeric_cols:
                    if pd.isna(df_filled.loc[idx, col]):
                        # Calcular promedio de otras estaciones en la misma fecha
                        other_cols = [c for c in numeric_cols if c != col]
                        available_data = df_filled.loc[idx, other_cols].dropna()
                        
                        if len(available_data) >= len(other_cols) * station_threshold:
                            # Usar factor de corrección basado en promedios históricos
                            col_mean = df_filled[col].mean()
                            others_mean = df_filled[other_cols].mean().mean()
                            
                            if not pd.isna(col_mean) and not pd.isna(others_mean) and others_mean > 0:
                                correction_factor = col_mean / others_mean
                                df_filled.loc[idx, col] = available_data.mean() * correction_factor
    
    else:
        # Métodos básicos como fallback
        return handle_missing_data(df_filled, method, max_gap)
    
    return df_filled

def assess_data_reliability_post_imputation(df_original, df_filled, data_type):
    """Evalúa la confiabilidad después de la imputación"""
    
    print(f"\n=== EVALUACIÓN POST-IMPUTACIÓN: {data_type.upper()} ===")
    
    numeric_cols = df_original.select_dtypes(include=[np.number]).columns
    
    reliability_scores = {}
    
    for col in numeric_cols:
        original_missing = df_original[col].isnull().sum()
        remaining_missing = df_filled[col].isnull().sum()
        total_records = len(df_original)
        
        # Calcular score de confiabilidad
        original_data_pct = ((total_records - original_missing) / total_records) * 100
        imputed_data_pct = ((original_missing - remaining_missing) / total_records) * 100
        
        # Score ponderado: datos originales tienen peso 1.0, imputados peso 0.7
        reliability_score = (original_data_pct * 1.0 + imputed_data_pct * 0.7) / 100
        
        reliability_scores[col] = {
            'original_data_pct': original_data_pct,
            'imputed_data_pct': imputed_data_pct,
            'remaining_missing_pct': (remaining_missing / total_records) * 100,
            'reliability_score': reliability_score
        }
        
        # Clasificar confiabilidad
        if reliability_score >= 0.85:
            reliability_class = "🟢 ALTA"
        elif reliability_score >= 0.7:
            reliability_class = "🟡 MEDIA"
        elif reliability_score >= 0.5:
            reliability_class = "🟠 BAJA"
        else:
            reliability_class = "🔴 MUY BAJA"
        
        print(f"  {col}:")
        print(f"    - Datos originales: {original_data_pct:.1f}%")
        print(f"    - Datos imputados: {imputed_data_pct:.1f}%")
        print(f"    - Aún faltantes: {(remaining_missing / total_records) * 100:.1f}%")
        print(f"    - Confiabilidad: {reliability_class} ({reliability_score:.2f})")
    
    # Confiabilidad general del dataset
    overall_reliability = np.mean([score['reliability_score'] for score in reliability_scores.values()])
    
    if overall_reliability >= 0.8:
        overall_class = "🟢 DATASET CONFIABLE"
        recommendation = "Proceder con análisis completo"
    elif overall_reliability >= 0.65:
        overall_class = "🟡 DATASET MODERADAMENTE CONFIABLE"
        recommendation = "Proceder con precaución, reportar limitaciones"
    elif overall_reliability >= 0.5:
        overall_class = "🟠 DATASET DE BAJA CONFIABILIDAD"
        recommendation = "Análisis limitado, buscar datos adicionales"
    else:
        overall_class = "🔴 DATASET NO CONFIABLE"
        recommendation = "No recomendado para análisis críticos"
    
    print(f"\n  EVALUACIÓN GENERAL: {overall_class}")
    print(f"  Confiabilidad promedio: {overall_reliability:.2f}")
    print(f"  Recomendación: {recommendation}")
    
    return reliability_scores, overall_reliability
    """
    Maneja datos faltantes usando diferentes estrategias
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame con datos faltantes
    method : str
        'interpolation', 'knn', 'forward_fill', 'backward_fill', 'mean'
    max_gap : int
        Máximo número de días consecutivos a interpolar
    """
    
    df_filled = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    print(f"\nAplicando método: {method.upper()}")
    
    for col in numeric_cols:
        original_missing = df_filled[col].isnull().sum()
        
        if method == 'interpolation':
            # Interpolación lineal con límite de gap
            df_filled[col] = df_filled[col].interpolate(
                method='linear', 
                limit=max_gap,
                limit_direction='both'
            )
            
        elif method == 'knn':
            # K-Nearest Neighbors para imputación
            if len(numeric_cols) > 1:  # Necesita múltiples variables
                try:
                    imputer = KNNImputer(n_neighbors=5)
                    df_filled[numeric_cols] = imputer.fit_transform(df_filled[numeric_cols])
                except:
                    # Fallback a interpolación si KNN falla
                    df_filled[col] = df_filled[col].interpolate(method='linear', limit=max_gap)
            else:
                # Fallback a interpolación si solo hay una columna
                df_filled[col] = df_filled[col].interpolate(method='linear', limit=max_gap)
                
        elif method == 'forward_fill':
            df_filled[col] = df_filled[col].fillna(method='ffill', limit=max_gap)
            
        elif method == 'backward_fill':
            df_filled[col] = df_filled[col].fillna(method='bfill', limit=max_gap)
            
        elif method == 'mean':
            # Imputación por promedio mensual/estacional
            if 'Fecha' in df_filled.columns:
                df_filled['mes'] = df_filled['Fecha'].dt.month
                monthly_means = df_filled.groupby('mes')[col].mean()
                
                for mes in range(1, 13):
                    mask = (df_filled['mes'] == mes) & (df_filled[col].isnull())
                    if mes in monthly_means.index and not pd.isna(monthly_means[mes]):
                        df_filled.loc[mask, col] = monthly_means[mes]
                
                df_filled.drop('mes', axis=1, inplace=True)
            else:
                df_filled[col] = df_filled[col].fillna(df_filled[col].mean())
        
        filled_missing = df_filled[col].isnull().sum()
        filled_count = original_missing - filled_missing
        
        print(f"  {col}: {filled_count} valores imputados, {filled_missing} aún faltantes")
    
    return df_filled

def create_missing_data_report(data):
    """Crea un reporte completo de datos faltantes"""
    
    print("\n" + "="*60)
    print("📊 REPORTE DE CALIDAD DE DATOS")
    print("="*60)
    
    quality_report = {}
    
    for key, df in data.items():
        if df is not None and not df.empty:
            missing_summary, missing_pct = assess_data_quality(df, key)
            quality_report[key] = {
                'summary': missing_summary,
                'overall_missing': missing_pct
            }
    
    # Visualización de datos faltantes
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Análisis de Datos Faltantes por Dataset', fontsize=16, fontweight='bold')
    
    datasets = list(data.keys())
    
    for i, (key, df) in enumerate(data.items()):
        if i >= 6:  # Máximo 6 subplots
            break
            
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        if df is not None and not df.empty:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            # Matriz de datos faltantes
            missing_matrix = df[numeric_cols].isnull()
            
            if not missing_matrix.empty and len(numeric_cols) > 0:
                sns.heatmap(missing_matrix.T, 
                           cbar=True, 
                           ax=ax, 
                           cmap='RdYlBu_r',
                           yticklabels=True,
                           xticklabels=False)
                ax.set_title(f'{key}\n({quality_report[key]["overall_missing"]:.1f}% faltantes)')
            else:
                ax.text(0.5, 0.5, 'Sin datos\nnuméricos', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(key)
        else:
            ax.text(0.5, 0.5, 'Dataset\nvacío', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(key)
    
    # Ocultar subplots no utilizados
    for i in range(len(data), 6):
        row = i // 3
        col = i % 3
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    return quality_report

def load_and_process_data_robust():
    """Carga y procesa datos con manejo robusto de errores y datos faltantes"""
    
    datasets = {
        'precip_papallacta': 'precipitacion_papallacta.csv',
        'precip_quijos': 'precipitacion_quijos.csv',
        'caudal_papallacta': 'caudal_papallacta.csv',
        'caudal_quijos': 'caudal_quijos.csv',
        'nivel_papallacta': 'nivel_papallacta.csv',
        'nivel_quijos': 'nivel_quijos.csv'
    }
    
    data = {}
    
    print("Cargando datasets...")
    
    for key, filename in datasets.items():
        try:
            df = pd.read_csv(filename, parse_dates=['Fecha'])
            
            # Agregar región para precipitación
            if 'precip' in key:
                region = key.split('_')[1].capitalize()
                df['region'] = region
            
            data[key] = df
            print(f"✅ {filename}: {len(df)} registros cargados")
            
        except FileNotFoundError:
            print(f"⚠️  {filename}: Archivo no encontrado")
            data[key] = None
        except Exception as e:
            print(f"❌ {filename}: Error al cargar - {e}")
            data[key] = None
    
    return data

def analyze_flow_contribution_robust(data):
    """Análisis de contribución con manejo optimizado de datos faltantes"""
    
    print("\n=== ANÁLISIS DE CONTRIBUCIÓN DE CAUDALES (OPTIMIZADO) ===\n")
    
    # Verificar disponibilidad de datos
    if data['caudal_papallacta'] is None or data['caudal_quijos'] is None:
        print("❌ Datos de caudal no disponibles para análisis")
        return None
    
    # Aplicar manejo optimizado de datos faltantes
    print("Procesando datos faltantes con estrategias optimizadas...")
    
    # Para Papallacta (más estaciones, más datos faltantes) - método híbrido
    caudal_pap_original = data['caudal_papallacta'].copy()
    caudal_pap = handle_missing_data_optimized(
        caudal_pap_original, 
        method='hybrid', 
        max_gap=5,
        station_threshold=0.5  # Al menos 50% de estaciones
    )
    
    # Evaluar confiabilidad post-imputación
    pap_reliability = assess_data_reliability_post_imputation(
        caudal_pap_original, caudal_pap, 'Caudal_Papallacta'
    )
    
    # Para Quijos (menos estaciones, mejor calidad) - método estacional
    caudal_qui_original = data['caudal_quijos'].copy()
    caudal_qui = handle_missing_data_optimized(
        caudal_qui_original, 
        method='seasonal', 
        max_gap=7
    )
    
    # Evaluar confiabilidad post-imputación
    qui_reliability = assess_data_reliability_post_imputation(
        caudal_qui_original, caudal_qui, 'Caudal_Quijos'
    )
    
    # Calcular totales con ponderación por confiabilidad
    caudal_pap_cols = ['H32', 'H34', 'H36', 'H45']
    available_cols_pap = [col for col in caudal_pap_cols if col in caudal_pap.columns]
    
    if available_cols_pap:
        # Calcular total con peso por confiabilidad de cada estación
        weights = []
        for col in available_cols_pap:
            if col in pap_reliability[0]:  # pap_reliability[0] contiene los scores
                weight = pap_reliability[0][col]['reliability_score']
                weights.append(weight)
            else:
                weights.append(0.5)  # peso por defecto
        
        # Total ponderado
        caudal_pap['total_caudal'] = 0
        for i, col in enumerate(available_cols_pap):
            caudal_pap['total_caudal'] += caudal_pap[col].fillna(0) * weights[i]
        
        # Normalizar por suma de pesos
        total_weight = sum(weights)
        if total_weight > 0:
            caudal_pap['total_caudal'] = caudal_pap['total_caudal'] / total_weight * len(available_cols_pap)
        
        # Calcular calidad de datos considerando disponibilidad y confiabilidad
        caudal_pap['data_quality'] = 0
        for i, col in enumerate(available_cols_pap):
            available_mask = ~caudal_pap[col].isnull()
            caudal_pap['data_quality'] += available_mask * weights[i]
        caudal_pap['data_quality'] = caudal_pap['data_quality'] / sum(weights)
        
    else:
        print("⚠️  No se encontraron columnas de caudal para Papallacta")
        return None
    
    # Caudal Quijos con confiabilidad
    if 'H33' in caudal_qui.columns:
        caudal_qui['total_caudal'] = caudal_qui['H33']
        
        # Asignar calidad basada en confiabilidad post-imputación
        qui_reliability_score = qui_reliability[0]['H33']['reliability_score'] if 'H33' in qui_reliability[0] else 0.7
        caudal_qui['data_quality'] = (~caudal_qui['H33'].isnull()).astype(float) * qui_reliability_score
        
    else:
        print("⚠️  No se encontró columna H33 para Quijos")
        return None
    
    # Filtrar datos con umbral ajustado por confiabilidad
    quality_threshold = 0.6  # Reducido debido a la alta cantidad de datos faltantes
    
    caudal_pap_clean = caudal_pap[caudal_pap['data_quality'] >= quality_threshold].copy()
    caudal_qui_clean = caudal_qui[caudal_qui['data_quality'] >= quality_threshold].copy()
    
    print(f"\nDatos de calidad después del filtrado optimizado:")
    print(f"- Papallacta: {len(caudal_pap_clean)}/{len(caudal_pap)} registros ({len(caudal_pap_clean)/len(caudal_pap)*100:.1f}%)")
    print(f"- Quijos: {len(caudal_qui_clean)}/{len(caudal_qui)} registros ({len(caudal_qui_clean)/len(caudal_qui)*100:.1f}%)")
    
    # Estadísticas con intervalos de confianza y porcentaje del mínimo
    def calculate_robust_stats_with_percentage(series, name, reliability_info=None):
        clean_series = series.dropna()
        if len(clean_series) == 0:
            print(f"{name}: Sin datos válidos")
            return 0
            
        mean_flow = clean_series.mean()
        percentage_of_minimum = (mean_flow / 127) * 100
        
        print(f"\n{name} (m³/s) - {len(clean_series)} registros válidos:")
        print(f"- Promedio: {mean_flow:.2f} ± {clean_series.std():.2f}")
        print(f"- Contribución al mínimo (127 m³/s): {percentage_of_minimum:.1f}%")
        
        if reliability_info:
            avg_reliability = reliability_info[1]  # overall_reliability
            print(f"- Confiabilidad del dataset: {avg_reliability:.2f} ({avg_reliability*100:.0f}%)")
        
        print(f"- Mediana: {clean_series.median():.2f}")
        print(f"- Percentil 25-75: {clean_series.quantile(0.25):.2f} - {clean_series.quantile(0.75):.2f}")
        print(f"- Rango: {clean_series.min():.2f} - {clean_series.max():.2f}")
        
        return percentage_of_minimum, mean_flow
    
    pap_percentage, pap_mean = calculate_robust_stats_with_percentage(
        caudal_pap_clean['total_caudal'], "CAUDALES PAPALLACTA", pap_reliability
    )
    qui_percentage, qui_mean = calculate_robust_stats_with_percentage(
        caudal_qui_clean['total_caudal'], "CAUDALES QUIJOS", qui_reliability
    )
    
    # Combinar datos con validación temporal y confiabilidad
    caudal_combined = pd.merge(
        caudal_pap_clean[['Fecha', 'total_caudal', 'data_quality']].rename(
            columns={'total_caudal': 'papallacta', 'data_quality': 'quality_pap'}
        ),
        caudal_qui_clean[['Fecha', 'total_caudal', 'data_quality']].rename(
            columns={'total_caudal': 'quijos', 'data_quality': 'quality_qui'}
        ),
        on='Fecha', how='outer'
    )
    
    # Calcular total con ponderación por confiabilidad
    pap_weight = pap_reliability[1]  # overall_reliability
    qui_weight = qui_reliability[1]  # overall_reliability
    
    mask_quality = (caudal_combined['quality_pap'].fillna(0) >= quality_threshold) & \
                   (caudal_combined['quality_qui'].fillna(0) >= quality_threshold)
    
    caudal_combined['total_afluentes'] = np.where(
        mask_quality,
        caudal_combined[['papallacta', 'quijos']].sum(axis=1, skipna=True),
        np.nan
    )
    
    # Agregar información de confiabilidad
    caudal_combined['reliability_score'] = np.where(
        mask_quality,
        (caudal_combined['quality_pap'].fillna(0) * pap_weight + 
         caudal_combined['quality_qui'].fillna(0) * qui_weight) / (pap_weight + qui_weight),
        np.nan
    )
    
    caudal_combined['cumple_minimo'] = caudal_combined['total_afluentes'] >= 127
    caudal_combined['data_reliability'] = mask_quality
    
    # Análisis de confiabilidad mejorado
    reliable_data = caudal_combined[caudal_combined['data_reliability']].copy()
    
    if len(reliable_data) > 0:
        total_mean = reliable_data['total_afluentes'].mean()
        total_percentage = (total_mean / 127) * 100
        avg_reliability = reliable_data['reliability_score'].mean()
        
        print(f"\nCAUDAL TOTAL AFLUENTES (m³/s) - {len(reliable_data)} registros válidos:")
        print(f"- Promedio: {total_mean:.2f} ± {reliable_data['total_afluentes'].std():.2f}")
        print(f"- CONTRIBUCIÓN AL MÍNIMO (127 m³/s): {total_percentage:.1f}%")
        print(f"- Confiabilidad promedio del análisis: {avg_reliability:.2f} ({avg_reliability*100:.0f}%)")
        print(f"- Mediana: {reliable_data['total_afluentes'].median():.2f}")
        print(f"- Rango: {reliable_data['total_afluentes'].min():.2f} - {reliable_data['total_afluentes'].max():.2f}")
        
        # Análisis crítico con consideración de confiabilidad
        registros_por_debajo = len(reliable_data[reliable_data['total_afluentes'] < 127])
        porcentaje_critico = (registros_por_debajo / len(reliable_data)) * 100
        
        print(f"\n=== ANÁLISIS DE IMPORTANCIA PARA COCA CODO SINCLAIR ===")
        print(f"🏭 Caudal mínimo requerido para operación: 127 m³/s")
        print(f"📊 CONTRIBUCIÓN PROMEDIO DE LOS AFLUENTES:")
        print(f"   • Papallacta: {pap_percentage:.1f}% del mínimo requerido (Confiabilidad: {pap_reliability[1]*100:.0f}%)")
        print(f"   • Quijos: {qui_percentage:.1f}% del mínimo requerido (Confiabilidad: {qui_reliability[1]*100:.0f}%)")
        print(f"   • TOTAL AFLUENTES: {total_percentage:.1f}% del mínimo requerido")
        
        # Clasificar importancia con ajuste por confiabilidad
        confidence_factor = min(avg_reliability, 1.0)
        adjusted_percentage = total_percentage * confidence_factor
        
        if adjusted_percentage >= 80:
            importance_level = "🟢 CRÍTICOS - MUY IMPORTANTES"
            importance_desc = f"Los afluentes son fundamentales ({total_percentage:.1f}% con {avg_reliability*100:.0f}% confiabilidad)"
        elif adjusted_percentage >= 60:
            importance_level = "🟡 MUY IMPORTANTES"
            importance_desc = f"Los afluentes son muy relevantes ({total_percentage:.1f}% con {avg_reliability*100:.0f}% confiabilidad)"
        elif adjusted_percentage >= 40:
            importance_level = "🟠 IMPORTANTES"
            importance_desc = f"Los afluentes tienen importancia significativa ({total_percentage:.1f}% con {avg_reliability*100:.0f}% confiabilidad)"
        elif adjusted_percentage >= 20:
            importance_level = "🔴 MODERADAMENTE IMPORTANTES"
            importance_desc = f"Los afluentes tienen importancia moderada ({total_percentage:.1f}% con {avg_reliability*100:.0f}% confiabilidad)"
        else:
            importance_level = "⚫ IMPORTANCIA LIMITADA"
            importance_desc = f"Los afluentes tienen importancia limitada ({total_percentage:.1f}% con {avg_reliability*100:.0f}% confiabilidad)"
        
        print(f"\n🎯 NIVEL DE IMPORTANCIA: {importance_level}")
        print(f"📝 INTERPRETACIÓN: {importance_desc}")
        
        print(f"\n📈 ESTADÍSTICAS OPERACIONALES:")
        print(f"- Registros analizados: {len(reliable_data)}")
        print(f"- Veces que NO alcanzan el mínimo: {registros_por_debajo} ({porcentaje_critico:.1f}%)")
        print(f"- Veces que SÍ alcanzan el mínimo: {len(reliable_data)-registros_por_debajo} ({100-porcentaje_critico:.1f}%)")
        print(f"- Confiabilidad del análisis: {len(reliable_data)/len(caudal_combined)*100:.1f}%")
        
        # Análisis de déficit con información de confiabilidad
        deficit_records = reliable_data[reliable_data['total_afluentes'] < 127].copy()
        if len(deficit_records) > 0:
            deficit_records['deficit'] = 127 - deficit_records['total_afluentes']
            print(f"\n⚠️  ANÁLISIS DE DÉFICIT (cuando no alcanzan 127 m³/s):")
            print(f"- Déficit promedio: {deficit_records['deficit'].mean():.2f} m³/s")
            print(f"- Déficit máximo: {deficit_records['deficit'].max():.2f} m³/s")
            print(f"- Déficit mínimo: {deficit_records['deficit'].min():.2f} m³/s")
            print(f"- Confiabilidad promedio en períodos déficit: {deficit_records['reliability_score'].mean():.2f}")
        
        # Evaluación de riesgo ajustada por confiabilidad
        reliability_factor = len(reliable_data)/len(caudal_combined)
        
        print(f"\n🚨 EVALUACIÓN DE RIESGO OPERACIONAL:")
        
        # Advertencia sobre limitaciones de datos
        if avg_reliability < 0.8:
            print(f"⚠️  LIMITACIÓN IMPORTANTE: Confiabilidad de datos {avg_reliability*100:.0f}%")
            print(f"    → Los resultados deben interpretarse con precaución")
            print(f"    → Se recomienda ampliar la red de monitoreo")
        
        if reliability_factor < 0.5:
            print("⚠️  DATOS INSUFICIENTES: Análisis requiere más información")
        elif porcentaje_critico > 20:
            print("⚠️  ALTO RIESGO: Los afluentes frecuentemente no aportan suficiente caudal")
            print("    → Se requieren fuentes adicionales o almacenamiento")
        elif porcentaje_critico > 10:
            print("⚠️  RIESGO MODERADO: Períodos críticos ocasionales")
            print("    → Monitoreo continuo y planes de contingencia")
        else:
            print("✅ RIESGO BAJO: Los afluentes generalmente aportan suficiente caudal")
            print("    → Situación operacional favorable")
            
    else:
        print("❌ No hay suficientes datos confiables para el análisis")
    
    return caudal_combined
    """Análisis de contribución con manejo de datos faltantes y cálculo de porcentajes"""
    
    print("\n=== ANÁLISIS DE CONTRIBUCIÓN DE CAUDALES (ROBUSTO) ===\n")
    
    # Verificar disponibilidad de datos
    if data['caudal_papallacta'] is None or data['caudal_quijos'] is None:
        print("❌ Datos de caudal no disponibles para análisis")
        return None
    
    # Aplicar manejo de datos faltantes
    print("Procesando datos faltantes en caudales...")
    
    caudal_pap = handle_missing_data(
        data['caudal_papallacta'].copy(), 
        method='interpolation', 
        max_gap=3
    )
    
    caudal_qui = handle_missing_data(
        data['caudal_quijos'].copy(), 
        method='interpolation', 
        max_gap=3
    )
    
    # Calcular totales con manejo de NaN
    caudal_pap_cols = ['H32', 'H34', 'H36', 'H45']
    available_cols_pap = [col for col in caudal_pap_cols if col in caudal_pap.columns]
    
    if available_cols_pap:
        caudal_pap['total_caudal'] = caudal_pap[available_cols_pap].sum(axis=1, skipna=True)
        # Marcar filas donde faltan demasiadas estaciones
        caudal_pap['data_quality'] = caudal_pap[available_cols_pap].count(axis=1) / len(available_cols_pap)
    else:
        print("⚠️  No se encontraron columnas de caudal para Papallacta")
        return None
    
    # Caudal Quijos
    if 'H33' in caudal_qui.columns:
        caudal_qui['total_caudal'] = caudal_qui['H33']
        caudal_qui['data_quality'] = (~caudal_qui['H33'].isnull()).astype(int)
    else:
        print("⚠️  No se encontró columna H33 para Quijos")
        return None
    
    # Filtrar datos de baja calidad
    quality_threshold = 0.7  # 70% de datos disponibles
    
    caudal_pap_clean = caudal_pap[caudal_pap['data_quality'] >= quality_threshold].copy()
    caudal_qui_clean = caudal_qui[caudal_qui['data_quality'] >= quality_threshold].copy()
    
    print(f"Datos de calidad después del filtrado:")
    print(f"- Papallacta: {len(caudal_pap_clean)}/{len(caudal_pap)} registros ({len(caudal_pap_clean)/len(caudal_pap)*100:.1f}%)")
    print(f"- Quijos: {len(caudal_qui_clean)}/{len(caudal_qui)} registros ({len(caudal_qui_clean)/len(caudal_qui)*100:.1f}%)")
    
    # Estadísticas con intervalos de confianza y porcentaje del mínimo
    def calculate_robust_stats_with_percentage(series, name):
        clean_series = series.dropna()
        if len(clean_series) == 0:
            print(f"{name}: Sin datos válidos")
            return 0
            
        mean_flow = clean_series.mean()
        percentage_of_minimum = (mean_flow / 127) * 100
        
        print(f"\n{name} (m³/s) - {len(clean_series)} registros válidos:")
        print(f"- Promedio: {mean_flow:.2f} ± {clean_series.std():.2f}")
        print(f"- Contribución al mínimo (127 m³/s): {percentage_of_minimum:.1f}%")
        print(f"- Mediana: {clean_series.median():.2f}")
        print(f"- Percentil 25-75: {clean_series.quantile(0.25):.2f} - {clean_series.quantile(0.75):.2f}")
        print(f"- Rango: {clean_series.min():.2f} - {clean_series.max():.2f}")
        
        return percentage_of_minimum
    
    pap_percentage = calculate_robust_stats_with_percentage(caudal_pap_clean['total_caudal'], "CAUDALES PAPALLACTA")
    qui_percentage = calculate_robust_stats_with_percentage(caudal_qui_clean['total_caudal'], "CAUDALES QUIJOS")
    
    # Combinar datos con validación temporal
    caudal_combined = pd.merge(
        caudal_pap_clean[['Fecha', 'total_caudal', 'data_quality']].rename(
            columns={'total_caudal': 'papallacta', 'data_quality': 'quality_pap'}
        ),
        caudal_qui_clean[['Fecha', 'total_caudal', 'data_quality']].rename(
            columns={'total_caudal': 'quijos', 'data_quality': 'quality_qui'}
        ),
        on='Fecha', how='outer'
    )
    
    # Calcular total solo cuando ambas fuentes tienen datos de calidad
    mask_quality = (caudal_combined['quality_pap'].fillna(0) >= quality_threshold) & \
                   (caudal_combined['quality_qui'].fillna(0) >= quality_threshold)
    
    caudal_combined['total_afluentes'] = np.where(
        mask_quality,
        caudal_combined[['papallacta', 'quijos']].sum(axis=1, skipna=True),
        np.nan
    )
    
    caudal_combined['cumple_minimo'] = caudal_combined['total_afluentes'] >= 127
    caudal_combined['data_reliability'] = mask_quality
    
    # Análisis de confiabilidad
    reliable_data = caudal_combined[caudal_combined['data_reliability']].copy()
    
    if len(reliable_data) > 0:
        total_mean = reliable_data['total_afluentes'].mean()
        total_percentage = (total_mean / 127) * 100
        
        print(f"\nCAUDAL TOTAL AFLUENTES (m³/s) - {len(reliable_data)} registros válidos:")
        print(f"- Promedio: {total_mean:.2f} ± {reliable_data['total_afluentes'].std():.2f}")
        print(f"- CONTRIBUCIÓN AL MÍNIMO (127 m³/s): {total_percentage:.1f}%")
        print(f"- Mediana: {reliable_data['total_afluentes'].median():.2f}")
        print(f"- Rango: {reliable_data['total_afluentes'].min():.2f} - {reliable_data['total_afluentes'].max():.2f}")
        
        # Análisis crítico mejorado con porcentajes de contribución
        registros_por_debajo = len(reliable_data[reliable_data['total_afluentes'] < 127])
        porcentaje_critico = (registros_por_debajo / len(reliable_data)) * 100
        
        print(f"\n=== ANÁLISIS DE IMPORTANCIA PARA COCA CODO SINCLAIR ===")
        print(f"🏭 Caudal mínimo requerido para operación: 127 m³/s")
        print(f"📊 CONTRIBUCIÓN PROMEDIO DE LOS AFLUENTES:")
        print(f"   • Papallacta: {pap_percentage:.1f}% del mínimo requerido")
        print(f"   • Quijos: {qui_percentage:.1f}% del mínimo requerido")
        print(f"   • TOTAL AFLUENTES: {total_percentage:.1f}% del mínimo requerido")
        
        # Clasificar importancia
        if total_percentage >= 100:
            importance_level = "🟢 CRÍTICOS - SUFICIENTES"
            importance_desc = "Los afluentes PUEDEN cubrir el mínimo requerido"
        elif total_percentage >= 80:
            importance_level = "🟡 MUY IMPORTANTES"
            importance_desc = "Los afluentes cubren la mayoría del caudal mínimo"
        elif total_percentage >= 50:
            importance_level = "🟠 IMPORTANTES"
            importance_desc = "Los afluentes cubren la mitad del caudal mínimo"
        elif total_percentage >= 25:
            importance_level = "🔴 MODERADAMENTE IMPORTANTES"
            importance_desc = "Los afluentes cubren una fracción del caudal mínimo"
        else:
            importance_level = "⚫ POCO IMPORTANTES"
            importance_desc = "Los afluentes aportan muy poco al caudal mínimo"
        
        print(f"\n🎯 NIVEL DE IMPORTANCIA: {importance_level}")
        print(f"📝 INTERPRETACIÓN: {importance_desc}")
        
        print(f"\n📈 ESTADÍSTICAS OPERACIONALES:")
        print(f"- Registros analizados: {len(reliable_data)}")
        print(f"- Veces que NO alcanzan el mínimo: {registros_por_debajo} ({porcentaje_critico:.1f}%)")
        print(f"- Veces que SÍ alcanzan el mínimo: {len(reliable_data)-registros_por_debajo} ({100-porcentaje_critico:.1f}%)")
        print(f"- Confiabilidad del análisis: {len(reliable_data)/len(caudal_combined)*100:.1f}%")
        
        # Análisis adicional: ¿Cuánto falta para llegar al mínimo?
        deficit_records = reliable_data[reliable_data['total_afluentes'] < 127].copy()
        if len(deficit_records) > 0:
            deficit_records['deficit'] = 127 - deficit_records['total_afluentes']
            print(f"\n⚠️  ANÁLISIS DE DÉFICIT (cuando no alcanzan 127 m³/s):")
            print(f"- Déficit promedio: {deficit_records['deficit'].mean():.2f} m³/s")
            print(f"- Déficit máximo: {deficit_records['deficit'].max():.2f} m³/s")
            print(f"- Déficit mínimo: {deficit_records['deficit'].min():.2f} m³/s")
        
        # Clasificación de riesgo ajustada por confiabilidad
        reliability_factor = len(reliable_data)/len(caudal_combined)
        
        print(f"\n🚨 EVALUACIÓN DE RIESGO OPERACIONAL:")
        if reliability_factor < 0.5:
            print("⚠️  DATOS INSUFICIENTES: Análisis requiere más información")
        elif porcentaje_critico > 20:
            print("⚠️  ALTO RIESGO: Los afluentes frecuentemente no aportan suficiente caudal")
            print("    → Se requieren fuentes adicionales o almacenamiento")
        elif porcentaje_critico > 10:
            print("⚠️  RIESGO MODERADO: Períodos críticos ocasionales")
            print("    → Monitoreo continuo y planes de contingencia")
        else:
            print("✅ RIESGO BAJO: Los afluentes generalmente aportan suficiente caudal")
            print("    → Situación operacional favorable")
    else:
        print("❌ No hay suficientes datos confiables para el análisis")
    
    return caudal_combined

def create_contribution_visualizations(data, caudal_combined):
    """Crea visualizaciones específicas del porcentaje de contribución al mínimo"""
    
    if caudal_combined is None or caudal_combined.empty:
        print("No hay datos para visualizar")
        return None
    
    # Filtrar datos confiables
    reliable_data = caudal_combined[caudal_combined['data_reliability']].copy()
    
    if len(reliable_data) == 0:
        print("No hay datos confiables para visualizar")
        return None
    
    # Calcular porcentajes de contribución
    reliable_data['pap_percentage'] = (reliable_data['papallacta'] / 127) * 100
    reliable_data['qui_percentage'] = (reliable_data['quijos'] / 127) * 100
    reliable_data['total_percentage'] = (reliable_data['total_afluentes'] / 127) * 100
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Análisis de Contribución al Caudal Mínimo (127 m³/s)\nHidroeléctrica Coca Codo Sinclair', 
                 fontsize=16, fontweight='bold')
    
    # 1. Gráfico de contribución por fuente (barras apiladas promedio)
    ax1 = axes[0, 0]
    
    pap_mean = reliable_data['pap_percentage'].mean()
    qui_mean = reliable_data['qui_percentage'].mean()
    total_mean = reliable_data['total_percentage'].mean()
    
    categories = ['Contribución\nPromedio']
    
    # Barras apiladas
    bar1 = ax1.bar(categories, [pap_mean], label=f'Papallacta ({pap_mean:.1f}%)', 
                   color='steelblue', alpha=0.8)
    bar2 = ax1.bar(categories, [qui_mean], bottom=[pap_mean], 
                   label=f'Quijos ({qui_mean:.1f}%)', color='lightcoral', alpha=0.8)
    
    # Línea del 100%
    ax1.axhline(y=100, color='red', linestyle='--', linewidth=2, 
                label='100% del mínimo requerido')
    
    ax1.set_ylabel('Porcentaje del Caudal Mínimo (%)')
    ax1.set_title(f'Contribución Promedio al Mínimo\nTotal: {total_mean:.1f}%')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(150, total_mean * 1.2))
    
    # Agregar texto con el total
    ax1.text(0, total_mean + 5, f'TOTAL: {total_mean:.1f}%', 
             ha='center', va='bottom', fontweight='bold', fontsize=12,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # 2. Serie temporal del porcentaje total
    ax2 = axes[0, 1]
    reliable_data_clean = reliable_data.dropna(subset=['total_percentage'])
    
    ax2.plot(reliable_data_clean['Fecha'], reliable_data_clean['total_percentage'], 
             color='darkgreen', alpha=0.7, linewidth=1.5, label='% del mínimo')
    ax2.axhline(y=100, color='red', linestyle='--', linewidth=2, 
                label='100% requerido')
    ax2.fill_between(reliable_data_clean['Fecha'], 0, 100, 
                     alpha=0.2, color='red', label='Déficit operacional')
    ax2.fill_between(reliable_data_clean['Fecha'], 100, reliable_data_clean['total_percentage'],
                     where=(reliable_data_clean['total_percentage'] >= 100),
                     alpha=0.3, color='green', label='Exceso disponible')
    
    ax2.set_title('Evolución Temporal de la Contribución')
    ax2.set_ylabel('Porcentaje del Caudal Mínimo (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Distribución de porcentajes
    ax3 = axes[1, 0]
    
    # Histograma del porcentaje total
    ax3.hist(reliable_data_clean['total_percentage'], bins=30, alpha=0.7, 
             color='skyblue', edgecolor='black', density=True)
    ax3.axvline(x=100, color='red', linestyle='--', linewidth=2, 
                label='100% mínimo')
    ax3.axvline(x=reliable_data_clean['total_percentage'].mean(), 
                color='green', linestyle='-', linewidth=2, label='Promedio')
    
    # Calcular estadísticas para el texto
    below_100 = len(reliable_data_clean[reliable_data_clean['total_percentage'] < 100])
    total_records = len(reliable_data_clean)
    pct_below = (below_100 / total_records) * 100
    
    ax3.set_title(f'Distribución de Contribución\n{pct_below:.1f}% del tiempo < 100%')
    ax3.set_xlabel('Porcentaje del Caudal Mínimo (%)')
    ax3.set_ylabel('Densidad')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Comparación por cuenca con porcentajes
    ax4 = axes[1, 1]
    
    # Crear datos para el boxplot
    pap_data = reliable_data['pap_percentage'].dropna()
    qui_data = reliable_data['qui_percentage'].dropna()
    
    bp = ax4.boxplot([pap_data, qui_data], 
                     labels=[f'Papallacta\n(Media: {pap_data.mean():.1f}%)', 
                            f'Quijos\n(Media: {qui_data.mean():.1f}%)'],
                     patch_artist=True)
    
    # Colorear las cajas
    colors = ['steelblue', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax4.axhline(y=50, color='orange', linestyle=':', linewidth=2, 
                label='50% del mínimo')
    ax4.axhline(y=100, color='red', linestyle='--', linewidth=2, 
                label='100% del mínimo')
    
    ax4.set_title('Contribución Individual por Cuenca')
    ax4.set_ylabel('Porcentaje del Caudal Mínimo (%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Crear un gráfico adicional tipo gauge/velocímetro
    fig2, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Crear gráfico de barras horizontal tipo gauge
    categories = ['Papallacta', 'Quijos', 'TOTAL AFLUENTES']
    percentages = [pap_mean, qui_mean, total_mean]
    colors = ['steelblue', 'lightcoral', 'darkgreen']
    
    bars = ax.barh(categories, percentages, color=colors, alpha=0.8)
    
    # Línea vertical del 100%
    ax.axvline(x=100, color='red', linestyle='--', linewidth=3, 
               label='100% del mínimo requerido (127 m³/s)')
    
    # Agregar valores en las barras
    for i, (bar, pct) in enumerate(zip(bars, percentages)):
        width = bar.get_width()
        ax.text(width + 2, bar.get_y() + bar.get_height()/2, 
                f'{pct:.1f}%', ha='left', va='center', fontweight='bold')
    
    ax.set_xlabel('Porcentaje del Caudal Mínimo Requerido (%)')
    ax.set_title('Importancia de los Afluentes para la Operación de Coca Codo Sinclair\n' + 
                f'(Caudal mínimo requerido: 127 m³/s)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(0, max(150, max(percentages) * 1.2))
    
    # Colorear el fondo según importancia
    if total_mean >= 100:
        ax.axvspan(100, ax.get_xlim()[1], alpha=0.1, color='green', 
                   label='Zona de suficiencia')
    if total_mean >= 80:
        ax.axvspan(80, 100, alpha=0.1, color='yellow')
    ax.axvspan(0, 80, alpha=0.1, color='red', label='Zona crítica')
    
    plt.tight_layout()
    plt.show()
    
    return fig, fig2

def generate_data_quality_recommendations(quality_report):
    """Genera recomendaciones específicas sobre calidad de datos"""
    
    print(f"\n=== RECOMENDACIONES DE CALIDAD DE DATOS ===\n")
    
    critical_datasets = []
    moderate_datasets = []
    
    for dataset, info in quality_report.items():
        missing_pct = info['overall_missing']
        
        if missing_pct > 30:
            critical_datasets.append((dataset, missing_pct))
        elif missing_pct > 15:
            moderate_datasets.append((dataset, missing_pct))
    
    if critical_datasets:
        print("🔴 DATASETS CRÍTICOS (>30% datos faltantes):")
        for dataset, pct in critical_datasets:
            print(f"   - {dataset}: {pct:.1f}% faltantes")
        print("   ACCIONES REQUERIDAS:")
        print("   - Búsqueda de fuentes de datos alternativas")
        print("   - Contacto con instituciones para datos históricos")
        print("   - Considerar exclusión temporal de períodos sin datos")
    
    if moderate_datasets:
        print("\n🟡 DATASETS MODERADOS (15-30% datos faltantes):")
        for dataset, pct in moderate_datasets:
            print(f"   - {dataset}: {pct:.1f}% faltantes")
        print("   ACCIONES SUGERIDAS:")
        print("   - Aplicar técnicas de imputación robustas")
        print("   - Validar resultados con análisis de sensibilidad")
    
    print(f"\n📋 ESTRATEGIAS GENERALES:")
    print("   1. Implementar control de calidad en tiempo real")
    print("   2. Establecer umbrales de aceptación de datos")
    print("   3. Documentar todas las transformaciones aplicadas")
    print("   4. Realizar análisis de sensibilidad con diferentes métodos")
    print("   5. Reportar limitaciones en conclusiones finales")

def generate_final_summary(caudal_combined):
    """Genera un resumen final con las conclusiones principales"""
    
    if caudal_combined is None:
        return
    
    reliable_data = caudal_combined[caudal_combined['data_reliability']].copy()
    
    if len(reliable_data) == 0:
        return
    
    # Calcular métricas clave
    total_mean = reliable_data['total_afluentes'].mean()
    total_percentage = (total_mean / 127) * 100
    pap_mean = reliable_data['papallacta'].mean()
    qui_mean = reliable_data['quijos'].mean()
    pap_percentage = (pap_mean / 127) * 100
    qui_percentage = (qui_mean / 127) * 100
    
    registros_por_debajo = len(reliable_data[reliable_data['total_afluentes'] < 127])
    porcentaje_critico = (registros_por_debajo / len(reliable_data)) * 100
    
    print(f"\n" + "="*80)
    print("📋 RESUMEN EJECUTIVO - IMPORTANCIA DE LOS AFLUENTES")
    print("="*80)
    
    print(f"\n🎯 PREGUNTA CLAVE: ¿Qué tan importantes son estos ríos para los 127 m³/s mínimos?")
    
    print(f"\n📊 RESPUESTA CUANTITATIVA:")
    print(f"   • Los afluentes analizados aportan {total_percentage:.1f}% del caudal mínimo requerido")
    print(f"   • Papallacta contribuye con {pap_percentage:.1f}% del mínimo")
    print(f"   • Quijos contribuye con {qui_percentage:.1f}% del mínimo")
    
    print(f"\n🏷️  CLASIFICACIÓN DE IMPORTANCIA:")
    if total_percentage >= 100:
        print("   🟢 CRÍTICOS Y SUFICIENTES - Pueden cubrir completamente el mínimo")
        impact = "ALTO"
    elif total_percentage >= 80:
        print("   🟡 MUY IMPORTANTES - Cubren la mayoría del caudal mínimo")
        impact = "ALTO"
    elif total_percentage >= 50:
        print("   🟠 IMPORTANTES - Cubren la mitad del caudal mínimo")
        impact = "MEDIO"
    elif total_percentage >= 25:
        print("   🔴 MODERADAMENTE IMPORTANTES - Fracción significativa")
        impact = "MEDIO-BAJO"
    else:
        print("   ⚫ POCO IMPORTANTES - Contribución mínima")
        impact = "BAJO"
    
    print(f"\n⚡ IMPACTO EN LA OPERACIÓN:")
    print(f"   • Nivel de impacto: {impact}")
    print(f"   • Frecuencia de déficit: {porcentaje_critico:.1f}% del tiempo")
    print(f"   • Caudal promedio aportado: {total_mean:.2f} m³/s de 127 m³/s requeridos")
    
    if porcentaje_critico > 20:
        risk_level = "ALTO"
        action = "Se requieren fuentes adicionales de agua"
    elif porcentaje_critico > 10:
        risk_level = "MODERADO"
        action = "Monitoreo continuo y planes de contingencia"
    else:
        risk_level = "BAJO"
        action = "Situación operacional favorable"
    
    print(f"\n🚨 NIVEL DE RIESGO: {risk_level}")
    print(f"   • Recomendación: {action}")
    
    print(f"\n📈 IMPLICACIONES PARA LA INVESTIGACIÓN:")
    if total_percentage >= 80:
        print("   ✅ Los afluentes son FUNDAMENTALES para la operación de la hidroeléctrica")
        print("   ✅ Cualquier afectación a estos ríos impactaría significativamente la generación")
        print("   ✅ Se justifica el monitoreo detallado y la protección de estas cuencas")
    elif total_percentage >= 50:
        print("   ⚠️  Los afluentes son IMPORTANTES pero no suficientes por sí solos")
        print("   ⚠️  Se requiere considerar otras fuentes o almacenamiento adicional")
        print("   ⚠️  El monitoreo es importante pero debe complementarse con otras medidas")
    else:
        print("   ℹ️  Los afluentes tienen importancia LIMITADA para la operación")
        print("   ℹ️  Otros factores y fuentes de agua son más críticos")
        print("   ℹ️  El foco de la investigación podría ampliarse a otras fuentes")
    
    print(f"\n💡 CONCLUSIÓN PRINCIPAL:")
    if total_percentage >= 100:
        conclusion = f"Los ríos analizados SON SUFICIENTES para garantizar el caudal mínimo"
    elif total_percentage >= 80:
        conclusion = f"Los ríos analizados son CRÍTICOS, aportando {total_percentage:.0f}% del mínimo"
    elif total_percentage >= 50:
        conclusion = f"Los ríos analizados son IMPORTANTES, aportando {total_percentage:.0f}% del mínimo"
    else:
        conclusion = f"Los ríos analizados tienen importancia LIMITADA ({total_percentage:.0f}% del mínimo)"
    
    print(f"   🎯 {conclusion}")
    
    print("="*80)

def main_robust():
    """Función principal con manejo robusto de datos faltantes"""
    
    print("🌊 ANÁLISIS ROBUSTO - HIDROELÉCTRICA COCA CODO SINCLAIR 🌊")
    print("=" * 70)
    
    # Cargar datos
    data = load_and_process_data_robust()
    
    # Evaluar calidad de datos
    quality_report = create_missing_data_report(data)
    
    # Generar recomendaciones de calidad
    generate_data_quality_recommendations(quality_report)
    
    # Análisis robusto de caudales
    if any(not df.empty for df in data.values()):
        caudal_combined = analyze_flow_contribution_robust(data)
        
        if caudal_combined is not None:
            # Crear visualizaciones de contribución
            print(f"\n🎨 Generando visualizaciones de contribución...")
            create_contribution_visualizations(data, caudal_combined)
            
            # Generar resumen final
            generate_final_summary(caudal_combined)
            
            print(f"\n✅ ANÁLISIS ROBUSTO COMPLETADO")
            print("📊 Se aplicaron técnicas de manejo de datos faltantes")
            print("🔍 Resultados incluyen evaluación de confiabilidad")
            print("📈 Se calcularon porcentajes de contribución al mínimo operacional")
        else:
            print(f"\n❌ No se pudo completar el análisis de caudales")
    else:
        print(f"\n❌ No se encontraron datos válidos para analizar")
    
    print("=" * 70)

if __name__ == "__main__":
    main_robust()
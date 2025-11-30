import pandas as pd 
import numpy as np 

df = pd.read_csv("EntregaTPFinal/Coursera.csv")
print("Primeras 3 filas del DataFrame: ")
print(df.head())
print("-----------")
print("\nInformación general del DataFrame:") 
df.info()
print("-----------")
print("\nEstadística descriptiva:")
print(df.describe(include='all'))
print("-----------")
print("Cantidad de valores nulos: ")
print(df.isnull().sum())
print("-----------")
#Limpiar la columna de texto reviewcount
df['reviewcount_num'] = df['reviewcount'].astype(str).str.replace('k', '*1000', regex=False)
def evaluar_expresion(x):

    if pd.isna(x) or x in ['None', 'nan', 'nan*1000']:
        return np.nan
    try:
        return eval(x)
    except:
        return np.nan 
df['reviewcount_num'] = df['reviewcount_num'].apply(evaluar_expresion)
df['reviewcount_num'] = df['reviewcount_num'].astype(float) 
mediana_rating = df['rating'].median()
mediana_reviews = df['reviewcount_num'].median()
df['rating'] = df['rating'].fillna(mediana_rating)
df['reviewcount_num'] = df['reviewcount_num'].fillna(mediana_reviews)
print("\n--- Verificación de Imputación de Nulos y Tipos ---")
print(f"Rating (Mediana): {mediana_rating:.2f}")
print(f"Reviewcount (Mediana): {mediana_reviews:.2f}")
print("Nulos restantes (debe ser 0):")
print(df[['rating', 'reviewcount_num']].isnull().sum())
#como ya esta limpia y creada una nueva, la original puedo eliminarla de la lista ya que podria causar confusiones
df = df.drop(columns=['reviewcount'])
print("\n--- Columnas restantes después de la limpieza final ---")
print(df.columns)
#Las columnnas de reviewcount y rating con valores vacios seran completadas con la mediana
mediana_rating = df['rating'].median()
mediana_reviews = df['reviewcount_num'].median()
df['rating'] = df['rating'].fillna(mediana_rating)
df['reviewcount_num'] = df['reviewcount_num'].fillna(mediana_reviews)
print("\n--- Verificación de Imputación de Nulos ---")
print(f"Mediana de Rating usada: {mediana_rating:.2f}")
print(f"Mediana de Reviewcount usada: {mediana_reviews:.2f}")
print(df[['rating', 'reviewcount_num']].isnull().sum())
#Las columnas que me quedan son tipos de datos categóricos por lo que los nulos serán reemplazados por la moda
df['skills']=df['skills'].fillna("No Especificado")
for col in ['level', 'certificatetype', 'duration']:
    moda = df[col].mode()[0]
    df[col] = df[col].fillna(moda)
print("\n--- Ver cantidad Nulos luego del rellenado")
print(df.isnull().sum())
print("-----------")
#Normalizacion de columnas
df['crediteligibility_norm'] = df['crediteligibility'].astype(int)
print("Normalización (crediteligibility)")
print(df['crediteligibility_norm'].head())
print("-----------")
print("Correccion tipos de datos (fechas, números, categorías).")
duration_mapping = {
    '1 - 3 Months': 3,
    '3 - 6 Months': 6,
    '6 - 12 Months': 12,
    'Less than 2 hours': 0, 
    '1 - 4 Weeks': 1,      
    '1 - 2 Weeks': 0.5,    
    '1 - 2 Months': 2,
    '2 - 4 Weeks': 1,
    '4 - 6 Months': 5,
    'Less than 1 hour': 0
}
df['duracion_meses'] = df['duration'].str.strip().map(duration_mapping)
df['duracion_meses'].fillna(df['duracion_meses'].median())
df = df.drop(columns=['duration'])
print("Verificación de 'duracion_meses'")
print(df[['duracion_meses']].head())
print(df['duracion_meses'].dtype)
print("-----------")
print("Calcular tasa de abandono general y por curso.")
total_cursos = len(df)
cursos_credito = df['crediteligibility_norm'].sum()
exito_general = (cursos_credito / total_cursos) * 100
abandono_general = 100 - exito_general
print(" porcentaje de Éxito y Abandono General")
print(f"Total de Cursos: {total_cursos}")
print(f"Cursos Elegibles para Crédito (Éxito): {cursos_credito}")
print(f"Tasa de Éxito General: {exito_general:.2f}%")
print(f"Tasa de Abandono General: {abandono_general:.2f}%")
exito_por_curso = df.groupby('course')['crediteligibility_norm'].mean().reset_index()
exito_por_curso['exito_%'] = exito_por_curso['crediteligibility_norm'] * 100
exito_por_curso['abandono_%'] = 100 - exito_por_curso['exito_%']
cursos_sin_exito = exito_por_curso.sort_values(
    by='abandono_%', 
    ascending=False
).head(5)
print("5 Cursos con Mayor Tasa de Abandono")
print(cursos_sin_exito)
# Calcular el promedio de meses de duración (horas de estudio)
promedio_duracion = df['duracion_meses'].mean()
print("Promedio de Horas de Estudio")
print(f"El promedio de duración de los cursos es de: {promedio_duracion:.2f} meses.")
df.to_csv('Coursera_Limpio_Final.csv', index=False)
print("-----------")
print("ARCHIVO FINAL GENERADO: 'Coursera_Limpio_Final.csv'")
print("-----------")
import seaborn as sns
import matplotlib.pyplot as plt
columnas_correlacion = [
    'crediteligibility_norm', 
    'rating',                 
    'reviewcount_num',        
    'duracion_meses'         
]
df_correlacion = df[columnas_correlacion]
matriz_correlacion = df_correlacion.corr()
print("Matriz de Correlación (Factores vs. Rendimiento")
print(matriz_correlacion)
plt.figure(figsize=(8, 7)) 
sns.heatmap(
    matriz_correlacion,
    annot=True,          
    cmap='coolwarm',     
    fmt=".2f",           
    linewidths=.5,       
    cbar_kws={'label': 'Coeficiente de Correlación'}
)
plt.title('Mapa de Calor de Correlación: Factores de Rendimiento')
plt.show()
plt.figure(figsize=(10, 6))
sns.boxplot(
    x='level', 
    y='rating', 
    data=df,
    palette='pastel'
)
plt.title('Distribución de Rating (Nota) por Nivel de Curso')
plt.xlabel('Nivel del Curso (Proxy de Grupo de Segmentación)')
plt.ylabel('Rating Promedio del Curso (Nota)')
plt.show()
plt.figure(figsize=(7, 6))
sns.countplot(
    x='crediteligibility_norm', 
    data=df,
    palette=['#ff9999', '#66b3ff'] 
)
plt.xticks([0, 1], ['Abandono (Sin Crédito)', 'Éxito (Con Crédito)'])
plt.title('Distribución de Cursos por Rendimiento (Éxito vs. Abandono)')
plt.xlabel('Rendimiento del Curso')
plt.ylabel('Conteo de Cursos')
plt.show()
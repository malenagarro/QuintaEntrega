#!/usr/bin/env python
# coding: utf-8

# # VIOLENCIA DE GÉNERO EN ARGENTINA
# 
# ## 1) INTRODUCCIÓN 
# ### CONTEXTO EMPRESARIAL
# A lo largo de los años, la situación vulnerable de la muejer en Argentna se está haciendo visible. Por lo que, exponer la cantidad de casos y sus efectos a toda la sociadas, contribuye a la visibilizacion de las mujeres que han sufrido y las que continuarán sufirendo a causa de la desgualdad de género. Además, permitiran avanzar pasos transcendentales en materia de políticas públicas en favor de la igualdad y contra las violencias de género. Generando acciones de corto, mediano y largo plazo sustentadas para la prevención, asistencia integral y protección de aquellas mujeres que atraviesan estas situaciones de violencia. Haciendo hincapie en aquellas. Por lo que, es muy importante analizar que edades son la que mayor cantidad de casos hay y en que provincias. La informacion obtenida corresponde a aquellas comunicaciones recibdad por la Línea 144, en donde las personas que se comunican acceden a dejar sus datos para un adecuado abordaje y seguimiento. Los registros corresponden a tres sedes: Provincia de Buenos Aires, CABA y la sede de gestión nacional. Las preguntas a responder son: - En que provincias se producen más casos? - Cuales son las edades en las que se produce más violencia?
# 
# ### CONTEXTO ANALÍTICO
# Los datos ya se han recopilado y están en uso:
# 1. El archivo ¨ViolenciaGenero2.0.xlsx" que contiene el historial de los casos de violencia de género en la Argentina desde el 2020.
# 2. El archivo "HabitantesProvincia.xlsx" que contiene la cantidad de habitantes por provincia que se determinó en el Censo 2022.
# 
# 
# ### OBJETIVOS 
# En este caso, se busca realizar un análisis estadístico y su consecuente compresión de los valores con el fin de determinar las provincias y edades más afectadas.Y, finalmente, crear un modelo para determinar los casos más probables por provincia, edad y vínculo con el agresor y, entonces, predecir los futuros casos y tipos de violencia. 
# 
# 
# Por lo tanto, se procederá de la siguiente manera: (1) se analizará los datos actuales y se evaluará las deficiencias; (2)extraer los datos de estas fuentes y realizar una limpieza de datos, EDA e ingeniería de características y (3) crear un modelo predictivo.

# ## 2) ANÁLISIS DE DATA EXISTENTE
# Antes de sumerginos en cualquier proyecto de ciencia de datos, siempre se debe evaluar los datos actuales para comprender que piezas de informacion podrían faltar. En algunos caso, no tendrá datos y tendrá que empezar de cero. En este caso, tenemos dos fuentes de datos diferentes, por lo que debemos analizar cada una de ellas individualmente y todas como un todo para averiguar como exactamente debemos complementarlas. En cada etapa, debemos tener en cuanta nuestro objetivo predecir futuros casos. Es decir, que debemos pensar la siguiente pregunta ¨Que información será útil pare predecir los futuros casos de violencia?¨

# ##### 2)1: IMPORTAMOS LIBRERIAS Y PAQUETES NECESARIAS

# In[1]:


import pandas                  as pd
from   scipy import stats
import numpy                   as np
import matplotlib.pyplot       as plt
import seaborn                 as sns
import statsmodels.formula.api as sm
import chart_studio.plotly     as py
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
# https://community.plot.ly/t/solved-update-to-plotly-4-0-0-broke-application/26526/2
import os


# ##### 2)2. IMPORTAMOS LA BASE DE DATOS QUE SE ENCUENTRA EN UN ARCHIVO EXCEL

# In[2]:


bd=pd.read_excel(r'C:\Users\garro\OneDrive\Escritorio\DATA SCIENCE\TRABAJO PRACTICO\ViolenciaGenero2.0.xlsx', sheet_name='casos')
bd2=pd.read_excel(r'C:\Users\garro\OneDrive\Escritorio\DATA SCIENCE\TRABAJO PRACTICO\HabitantesProvincia.xlsx', sheet_name='cantidad')


# ##### 2) 3. VERFICAMOS SI SE REALIZÓ LA CARGA A PARTIR DE M0STRAR LOS PRIMEROS CINCO DATOS

# In[3]:


bd.head()


# In[4]:


bd2.head()


# ##### 2) 4. INSPECCIONAMOS EL DATASET PARA COMPRENDER LOS DATOS QUE TENEMOS

# In[5]:


bd.dtypes


# ## 3) LIMPIEZA Y TRANSFORMACIÓN DE DATOS

# ##### 3) 1. CAMBIAMOS EL FORMATO DE LA COLUMNA FECHA A 'DATETIME' 

# In[6]:


bd['FECHA']=pd.to_datetime(bd.FECHA, errors='coerce')
bd.head()


# ##### 3) 2. VISUALIZACIÓN DE LOS OUTLIERS Y REMOCIÓN
# ###### 3) 2.A: VISUALIZACIÓN A TRAVÉS DEL BOXPLOT

# In[7]:


ax=sns.boxplot(x='EDAD', data=bd)


# ###### 3) 2.B: CÁLCULO ANALÍTICO (puede ser también con la función describe())

# In[8]:


Q1=bd['EDAD'].quantile(0.25)
print('Primer cuartil', Q1)

Q3=bd['EDAD'].quantile(0.75)
print('Tercer cuartil', Q3)

IQR=Q3-Q1
print('Rango intercuartil', IQR)

mediana=bd['EDAD'].median()
print('mediana', mediana)

valor_min=bd['EDAD'].min()
print('Valor mínimo', valor_min)

valor_max=bd['EDAD'].max()
print('Valor máximo', valor_max)

Valor_BI=(Q1-1.5*IQR)
print('Valor_BI', Valor_BI)

Valor_BS=(Q3+1.5*IQR)
print('Valor_BS', Valor_BS)


# ###### 3) 2.C: ELIMINACIÓN DE OUTLIERS
# Con esta función, se detecta el outlier y se intercambia con el valor de la media.

# In[9]:


def clean_age(age):
    if age>=Valor_BI and age<=Valor_BS:
        return age
    else:
            return mediana
bd['age_clean'] = bd['EDAD'].apply(clean_age)

# Check out the new column and make sure it looks right

print("'EDADES'")
print("Valor mínimo: ", bd["age_clean"].min())
print("Valor máximo: ", bd["age_clean"].max())


# ###### 3) 2.D: VISUALIZACIÓM DE DATOS SIN ULIERS

# In[10]:


ax=sns.boxplot(x='age_clean', data=bd)


# ##### 3) 3. LIMPIEZA DE VALORES NaN. REEMPLAZO DE NaN POR MEDIA

# In[11]:


bd['age_clean'].fillna(value=mediana, inplace=True)
bd.head(10)


# ## 4) VISUALIZACIÓN DE LOS DATOS Y BÚSQUEDA DE PATRONES
# ##### 4) 1. VISUALIZACIÓN DE LA VARIACIÓN DE CANT DE CASOS EN EL TIEMPO
# ###### 4) 1.A: PARA FACILITAR EL ANÁLISIS AGRUPAMOS LAS FECHAS DE MANERA MENSUAL (no diaria)

# In[12]:


bd['FECHA'] = pd.to_datetime(bd['FECHA'], errors='coerce')
bd['FECHA_MES']=bd.FECHA.dt.to_period('M')
bd3=bd.groupby('FECHA_MES', as_index=False).sum()
bd3.head()


# In[13]:


bd3['FECHA_MES'] = bd3['FECHA_MES'].astype('str')


# ###### 4) 1.B: REALIZAMOS UN GRÁFICO DE LINEAS

# In[14]:


plt.figure(figsize=(12,10))
plt.plot(bd3.FECHA_MES, bd3.CASO)
plt.xticks(rotation = 'vertical')
plt.ylabel('Cantidad de casos')
plt.title('Variación en el tiempo')
plt.xlabel('Tiempo')
plt.show()


# ##### 4) 3. REALIZAMOS UN HISTOGRAMA EN EL QUE PODAMOS ANALIZAR LA DISTRIBUCIÓN DE LAS EDADES EN CADA PROVINCIA

# In[15]:


sns.displot(data=bd, x="EDAD", hue="PROVINCIA", multiple="stack")


# ##### 4) 4. GRÁFICAMOS CANTIDAD DE CASOS POR PROVINCIA:
# ###### 4) 4.A: CALCULAMOS LA CANTIDAD DE CASOS TOTALES POR PROVINCIA

# In[16]:


serie_provincia=bd.PROVINCIA.value_counts()
serie_provincia


# ###### 4) 4.B: GRAFICAMOS

# In[17]:


fig, ax= plt.subplots()
ax.barh(serie_provincia.index, serie_provincia, label='Casos totales')
ax.legend(loc='upper right')
ax.set_title('Cantidad de casos en cada provincia')
ax.set_ylabel('Provincias')
ax.set_xlabel('Cantidad de casos')


# ###### 4) 4.C: PARA MEJORAR LA VISUALIZACIÓN PODEMOS REALIZAR UNA CATEGORIZACION DE LAS PROVINCIAS EN REGIONES

# In[18]:


# REALIZAMOS UNA COPIA DE LA COLUMNA PROVINCIA PARA PRESERVAR LOS DATOS ORIGNALES.
bd['REGION'] = bd['PROVINCIA']

# EN LA NUEVA COLUMNA ASIGNAMOS UNA NUEVA CATEGORIA
PAMPEANA = ['Ciudad Autónoma de Buenos Aires', 'Buenos Aires', 'Córdoba', 'Entre Ríos', 'La Pampa','Santa Fe']
NOA = ['Catamarca', 'Jujuy', 'La Rioja', 'Salta', 'Santiago del Estero', 'Santiago Del Estero', 'Tucumán']
NEA = ['Corrientes', 'Chaco', 'Formosa', 'Misiones'] 
CUYO = ['Mendoza', 'San Luis', 'San Juan']
PATAGONIA = ['Chubut', 'Neuquén', 'Río Negro', 'Santa Cruz', 'Tierra del Fuego, Antártida e Islas del Atlántico Sur']

bd['REGION'] = bd['REGION'].apply(lambda x:"PAMPEANA" if x in PAMPEANA else x)
bd['REGION'] = bd['REGION'].apply(lambda x:"NOA" if x in NOA else x)
bd['REGION'] = bd['REGION'].apply(lambda x:"NEA" if x in NEA else x)
bd['REGION'] = bd['REGION'].apply(lambda x:"CUYO" if x in CUYO else x)
bd['REGION'] = bd['REGION'].apply(lambda x:"PATAGONIA" if x in PATAGONIA else x)

# CREAMOS UNA SERIE DE LAS REGIONAS CONTANDO LOS CASOS
serie_regiones=bd.REGION.value_counts()
serie_regiones

print(serie_regiones)


# ###### 4) 4.D: GRAFICAMOS

# In[19]:


fig, ax= plt.subplots()
ax.barh(serie_regiones.index, serie_regiones, label='Casos totales')
ax.legend(loc='upper right')
ax.set_title('Cantidad de casos en cada provincia')
ax.set_ylabel('Cantidad de casos')
ax.set_xlabel('Regiones')


# ##### 4) 5. REALIZAMOS UNA ESTADICTICA DE EDADES POR PROVINCIA

# In[20]:


sns.boxplot(x=bd.REGION, y= bd.EDAD)
plt.title('Boxplot comparativo entre las regiones de Argentina en funcion de la edad')
plt.xlabel('Región')
plt.ylabel('Edad')


# Por lo tanto, gracias a estos gráficos podemos realizar las primeras conclusiones:
# 
# * la mayor cantidad de casos se da en casos con victimas de un rango de edad de entre 27-43 años, siendo la edad en la que se concentra el 50% de los casos a los 34 años
# * la provincia con mayor cantidad de casos es Buenos Aires, mientras que la de menor cantidad es La Pampa
# 
# AVISO IMPORTANTE! PARA UNA MAYOR CARACTERIZACIÓN DE LOS DATOS DEBEMOS REALIZAR UNA NORMALIZACIÓN DE ESTOS. A partir de dividir esta cantidad de casos por provincia por su respectiva cantidad de habitantes (queda pendiente para la proxima entrega)

# In[21]:


model1 = 'EDAD~REGION'
lm1   = sm.ols(formula = model1, data = bd).fit()
print(lm1.summary())


# ##### 4) 6. ANALIZAMOS EL VINCULO DEL AGRESOR CON LA VICTIMA
# ###### 4) 6.A: REALIZAMOS UN RECUENTO DE CASOS POR AGRESOR

# In[22]:


vinculo=bd.groupby('VINCULO_PERSONA_AGRESORA')
cant=bd.groupby(bd.VINCULO_PERSONA_AGRESORA)['CASO'].count()
cant


# ###### 4) 6.B: REALIZAMOS UN GRÁFICO DE TORTA PARA VISUALIZACIÓN DE DATOS

# In[23]:


fig1, ax1 = plt.subplots()
#Creamos el grafico, añadiendo los valores
vinculo=['Ex pareja', 'Madre o tutor', 'Otro', 'Otro familiar', 'Padre o tutor', 'Pareja', 'Superior jerárquico']
ax1.pie(cant, labels=vinculo, autopct='%1.1f%%', shadow=True, startangle=90)
#señalamos la forma, en este caso 'equal' es para dar forma circular
ax1.axis('equal')
plt.title('Distribución de vinculo de agresor')
#plt.legend()
plt.savefig('grafica_pastel.png')
plt.show()


# ###### 4) 6.C: REALIZAMOS HISTOGRAMA QUE ANALICE EL VINCULO DEL AGRESOR EN FUNCIÓN A LA EDAD DE LA VICTIMA
# ESTO PERMITIRA CONOCER PARA CUALES SON LAS EDADES MAS VULNERABLES PARA CADA TIPO DE ÄGRESOR

# In[24]:


plt.figure()
# Figure -level
ax = sns.displot(data=bd, x='EDAD', kind='kde', hue='VINCULO_PERSONA_AGRESORA', fill=True)
ax.set(xlabel='Edad', ylabel='Densidad', title='Distribución  de edades en función a vincuo con agresor')


# ###### 4) 6.D: GRAFICAMOS BOXPLOT PARA ANLIZAR LAS EDADES

# In[25]:


sns.boxplot(x=bd.VINCULO_PERSONA_AGRESORA, y= bd.EDAD)
plt.title('Boxplot comparativo vinculo de persona en funcion de la edad')
plt.xlabel('Vinculo de persona agresora')
plt.ylabel('Edad')


# In[26]:


model2 = 'EDAD~VINCULO_PERSONA_AGRESORA'
lm1   = sm.ols(formula = model2, data = bd).fit()
print(lm1.summary())


# ## 5) FEATURE SELECTION
# ### 5) 1. METODO DE FILTRO
# Como primer instancia usaremos el método de filtro para tener una idea general de que variables son las más importantes. 

# Para iniciar podemos verificar todas las corelaciones posibles a partir de instalar la libreria "Pandas_profiling" que permite generar un reporte del DataFrame.

# In[27]:


get_ipython().system(' pip install https://github.com/pandas-profiling/pandas-profiling/archive/master.zip')
import pandas_profiling


# In[28]:


profile = pandas_profiling.ProfileReport(bd)
profile


# Por lo tanto, como nuestro objetivo es predecir la cantidad de casos y el tipo de violencia, debemos hacer una limpieza de ciertas variabes. Donde, nos quedaremos con los tipos de violencia, el vínculo del agresor con la victima y la nueva columna generada de las edades sin outliers y NaN (age_clean).

# ##### 5)1.A. VARIABLES NUMERICAS

# In[29]:


bd2=bd.drop(['FECHA','PROVINCIA','PAIS_NACIMIENTO', 'GENERO_PERSONA_SIT_VIOLENCIA','GENERO_AGRESOR', 'EDAD', 'REGION','FECHA_MES', 'CASO'], axis = 1) 
bd2


# In[30]:


bd3=pd.get_dummies(bd2, columns=['VINCULO_PERSONA_AGRESORA'])
bd3


# In[31]:


correlacion=bd3.corr()
correlacion


# In[32]:


cortarget=abs(correlacion)
relevantfeatures= cortarget[cortarget>=0.05]
plt.figure(figsize=(12, 6))
heatmap=sns.heatmap(cortarget[abs(correlacion)>=0.05][[x for x in relevantfeatures.index]], annot=True, cmap=plt.cm.Reds)
plt.xticks(rotation=45) 
plt.show()


# Como podemos observar que no existe una gran correlación entre los tipos de violencia y el vinculo de la víctima con el agrsor.  

# ##### 5)1.B: VARIABLES CATEGORICAS

# In[33]:


serie_contingencia=pd.crosstab(bd.VINCULO_PERSONA_AGRESORA, bd.age_clean)
serie_contingencia


# Por lo tanto, al no existir una gran correlación entre las variables decidimos disminuir la dimensionalidad seleccionando las varibales que consideramos más relevantes para el análisis:
# 
# 1. age_clean (edad de la víctima)
# 2. VINCULO_PERSONA_AGRESORA: el vinculo que tiene la vinctima con la persona agresora ("pareja", ëx pareja". "padre o tutor", "madre o tutor", "otro famiiar", "superior jerárquico", "otro")
# 3. todas las columnas pertenecientes a los tipos de violencia ('TIPO_VIOLENCIA_FISICA', 'TIPO_VIOLENCIA_PSICOLOGICA', 'TIPO_VIOLENCIA_SEXUAL' , 'TIPO_VIOLENCIA_ECONOMICA', 'TIPO_VIOLENCIA_SIMBOLICA', 'TIPO_VIOLENCIA_DOMESTICA',	'TIPO_VIOLENCIA_INSTITUCIONAL', 'TIPO_VIOLENCIA_LABORAL', 'TIPO_VIOLENCIA_CONTRA_LIBERTAD_REPRODUCTIVA', 'TIPO_VIOLENCIA_OBSTETRICA', 'TIPO_VIOLENCIA_MEDIATICA',  'TIPO_VIOLENCIA_OTRAS')

# ### 5) 2. METODO DE ENVOLTURA  (Forward Selection)
# Ahora usaremos el forward selection. Debido a su complejidad, será con el que que nos quedaremos con los resultados.
# 
# ##### 5) 2.1. AISLAMOS LAS VARIABLES DE INTERES Y LAS INDEPENDIENTES

# In[34]:


y = bd3['TIPO_VIOLENCIA_FISICA']
X = bd3.drop('TIPO_VIOLENCIA_FISICA', axis=1)


# ##### 5) 2.2. REALIZAMOS EL PRIMER SPRINT
# En este sprint se seleccionaran 5 features.

# In[35]:


#Librerias
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LinearRegression
# Sequential Forward Selection(sfs)
sfs1 = SFS(LinearRegression(),
          k_features=5,
          forward=True,
          floating=False,
          scoring = 'r2',
          cv = 0)


# In[36]:


sfs1.fit(X, y)
sfs1.k_feature_names_  


# ##### 5) 2.3. REALIZAMOS EL SEGUNDO SPRINT
# En este sprint se seleccionaran 8 features.

# In[37]:


#Librerias
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LinearRegression
# Sequential Forward Selection(sfs)
sfs2 = SFS(LinearRegression(),
          k_features=8,
          forward=True,
          floating=False,
          scoring = 'r2',
          cv = 0)


# In[38]:


sfs2.fit(X, y)
sfs2.k_feature_names_ 


# ## 6) MODELAMIENTO
# ### 6) 1. ARBOL DE DECISION
# Son estructuras matemáticas (diagramas de flujo) que utilizan criterios de teoría de la información como la impureza para hacer segmentaciones
# 

# ###### 6)1.1. IMPORTAMOS LIBRERIAS

# In[47]:


from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree


# In[58]:


## y = bd3['TIPO_VIOLENCIA_FISICA']
X = bd3.drop('TIPO_VIOLENCIA_FISICA', axis=1)

# ajustar arbol de decisión simple con hiperparametros (defecto)
clf = DecisionTreeClassifier(random_state=1234, max_depth=2)
model = clf.fit(X, y)
# Graficando
fig = plt.figure(figsize=(18,12))
_ = tree.plot_tree(clf,feature_names=X.columns,  
                   class_names=y.unique().astype('str'),
                   filled=True)


# ### 6) 2. LINEAR REGRESSION
# 
# ##### 6) 2. 1. PRIMER SPRINT
# Aplicamos el modelo de regresión lineal para predecir el tipo de violencia con las cinco variables seleccionadas en el primer sprint. 
# 
# ###### 6)2. 1. A. SPLIT DATA AND DATA TRAIN

# In[40]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X[list(sfs1.k_feature_names_)], y, test_size=0.30, random_state=42) 


# ###### 6)2.1.B. AJUSTE DEL MODELO

# In[41]:


#ajuste de una regresion lineal utilizando statsmodels 
import statsmodels.api as sm 
#Variables utilizar para nuestra regresion lo obtenido anteriormente con el facture 
X_1sprint1= X_train

#vamos a tomar una columna de constantes para ajustar nuestro modelo 
x_constante1= sm.add_constant(X_1sprint1)
#genera el modeolo 
li_reg=sm.OLS(y_train, x_constante1).fit()
li_reg.summary()


# ###### 6) 2. 1. C. METRICAS

# In[42]:


from sklearn.metrics import mean_squared_error, r2_score 
Xnuevo1=sm.add_constant(X_test)
pred_test1=li_reg.predict(Xnuevo1)
print('Coeficiente de determinacion')
print('R2 test model', r2_score(y_test, pred_test1))
print("-"*50)
print('Mean squared error')
print('MSE test model', mean_squared_error(y_test, pred_test1))
print("-"*50)


# ###### 6)2.1.D. CONCLUSIONES
# El coeficiente de determinación es bastante bajo, lo cual nos indica que el modelo está muy mal ajustado.
# 
# Por otro lado, el error cuadrático medio es un valor único que proporciona información sobre la bondad del ajuste de la línea de regresión. Cuanto menor sea el valor de MSE, mejor será el ajuste, ya que los valores más pequeños implican menores magnitudes de error. Es decir, como es menor a 1, los puntos están bastante proximos a los valores predichos del modelo, lo cual nos va a dar que el modelo predice con buena exactitud.
# 
# ##### 6) 2.2. SEGUNDO SRINT
# Se utilizará un modelo de regresión lineal para predecir los gastos totales con las 8 variables seleccionadas en el segundo sprint
# ###### 6)2. 2. A. SPLIT DATA AND DATA TRAIN

# In[44]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X[list(sfs2.k_feature_names_)], y, test_size=0.30, random_state=42) 


# ###### 6)2. 2. B. AJUSTE DEL MODELO

# In[45]:


#ajuste de una regresion lineal utilizando statsmodels 
import statsmodels.api as sm 
#Variables utilizar para nuestra regresion lo obtenido anteriormente con el facture 
X_1sprint2= X_train

#vamos a tomar una columna de costantes para ajustar nuestro modelo 
x_constante2= sm.add_constant(X_1sprint2)
#genera el modeolo 
li_reg=sm.OLS(y_train, x_constante2).fit()
li_reg.summary()


# ###### 6)2. 2. C. METRICAS

# In[46]:


from sklearn.metrics import mean_squared_error, r2_score 
Xnuevo2=sm.add_constant(X_test)
pred_test2=li_reg.predict(Xnuevo2)
print('Coeficiente de determinacion')
print('R2 test model', r2_score(y_test, pred_test2))
print("-"*50)
print('MEan squared error')
print('MSE test model', mean_squared_error(y_test, pred_test2))
print("-"*50)


# ###### 6)2.2.D. CONCLUSIONES
# El coeficiente de determinación continua siendo bastante bajo y se nota una mejoría leve con respecto al del primer sprint con 5 variables, lo cual nos indica que el modelo está mejor ajustado, obviamente que no de manera perfecta pero sí con una mayor eficiencia.
# 
# Además, también podemos observar una leve mejoría en el error cuadrático medio, ya que disminuye

# #### 6) 3. REGRESION LOGÍSTICA
# ###### 6)3.1. IMPORTAMOS LIBRERIAS

# In[59]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# ###### 6)3.2. SPLIT DATA AND TRAIN DATA 

# In[60]:


X_train, X_test, y_train, y_test = train_test_split(X, y)
model = LogisticRegression(max_iter=10000, n_jobs=-1) 


# ###### 6)3.3. AJUSTE DE MODELO Y PREDICCIONES

# In[63]:


# Separacion train/tet
X_train, X_test, y_train, y_test = train_test_split(X, y)
model = LogisticRegression(max_iter=10000, n_jobs=-1) 
# Ajustar modelo
model.fit(X_train, y_train) 
#Predicciones
predicciones = model.predict(X_test)
predicciones

print(accuracy_score(y_test, predicciones))


# ###### 6)3.4. REALIZAMOS MATRIZ DE CONFUSION
# Esta matriz es una herramienta que permite visualizar el desempeño de un algoritmo  de aprendizaje supervisado. Cada columna de la matriz representa el número de predicciones de cada clase, mientras que cada fila representa a las instancias en la clase real., o sea en términos prácticos nos permite ver  qué tipos de aciertos y errores está teniendo nuestro modelo a la hora de pasar por el proceso de aprendizaje con los datos.

# In[64]:


from sklearn.metrics import confusion_matrix
#Matriz de confusion
cf_matrix = confusion_matrix(y_test, predicciones)
import seaborn as sns
ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')
ax.set_title('Matriz de confusion con labels\n\n');
ax.set_xlabel('\nValores predichos')
ax.set_ylabel('Valores reales ');
## Ticket labels - En orden alfabetico
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])
plt.show()


# ## CONCLUSIONES
# Usamos las técnicas de correlación para determinar si existía o no una relaciones entre diferentes variables
# en los casos de violencia de género. 
# En primer lugar, buscamos determinar la cantidad de casos por provincia y cual es su varianza. Esto nos permite definir
# cuales son los rangos de edad más vulnerables y en un futuro, enfatizar las acciones legislativas y de asistencia a 
# esas edades. En este caso, observamos que el 50% de los casos se acumulaban entre los 27 (25%) a 43 (75%) años, siendo
# el 50% los 34 años.
# En segundo lugar, realizamos un gráfico de barras que indicaba la cantidad de casos por provincia. Esto permite darnos 
# una idea de cuales son las provincias más vulerables y, por lo tanto, crear una mayor cantidad de centros de asistencia
# a la mujer. En este caso, observamos que la provincia con mayor cantidad de casos es Buenos Aires,seguida por la provincia de Santa Fe, mientras que la de menor cantidad de asos es La Pampa. 
# Luego, para facilitar la visualización de los datos realizamos una categorización de las provincias en regiones (Pampeana,
# NOA, NEA, Cuyo y Patagonia).Además, buscamos identificar la media de las edades de las victimas nn cada región. 
# En tercer lugar, determinamos la cantidad de casos en función al vinculo del agresor con la víctima y, cuales son las edades
# más afectadas en cada tipo de vínculo. Observamos que las victimas cuyo agresor es un familiar (madre, padre, tutor u ptro
# familiar) son entre los 11 a 20 años. Mientras que, aquellas en las que el agresor es la pareja o ex pareja son de entre 18 
# a 45 años. 
# A partir de estas conclusiones, se recomienda:
# - tanto en las provincias de Buenos Aires, Santa Fe, Tucumán, Mendoza y Ciudad de Buenos Aires incrementar la cantidad de centros de ayuda para la mujer
# - debido a que la media de casos en la región del NOA y NEA son las regiones con una mayor cantidad de casos en menores, se recomienda la creación de talleres en colegios, para fomentar la comunicación de los alumnos con el cuerpo de profesores cuando exista algún caso de violencia. 
# 
# Por último, se hicieron pruebas de tres modelos de regresión que nos permitirán hacer las predicciones a futuros casos de violencia. Siendo el de mejor performance la regresión logística con una exactitud de 70%.

# In[ ]:





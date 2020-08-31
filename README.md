# TFG-Estadistica
Scripts utilizados para mi proyecto final de grado.

En el directorio principal estan los archivos jupyter notebooks para visualizar los datos (con reducción de dimensionalidad) y para hacer clustering en dos fases (two-stages clustering). Hay uno para cada dataset (MNIST, Fashion, Cifar).

En la carpeta DCEC estan el archivos utilizados para utilizar el algoritmo DCEC de deep clustering:
DCEC.ipynb es el script principal. Desde allí se importan algunas funciones de Varios.py (gráficos, métricas, autoencoders etc.), de DCEC.py se importa el algoritmo y desde Datasets los datos. En la carpeta 'networks' se encuentran guardas las redes neuronales ya entrenadas. 

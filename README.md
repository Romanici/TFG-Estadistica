# TFG-Estadistica
Scripts utilizados para mi proyecto final de grado.

En el directorio principal estan los archivos jupyter notebooks para visualizar los datos (con reducción de dimensionalidad) y para hacer clustering en dos fases (two-stages clustering). Hay uno para cada dataset (MNIST, Fashion, Cifar).

En la carpeta DCEC estan el archivos utilizados para utilizar el algoritmo DCEC de deep clustering:
DCEC.ipynb es el script principal. Desde allí se importan algunas funciones de Varios.py (gráficos, métricas, autoencoders etc.), de DCEC.py se importa el algoritmo y desde Datasets los datos. En la carpeta 'networks' se encuentran guardas las redes neuronales ya entrenadas. 

---

Scripts used for my final degree project.

In the main directory you can find the jupyter notebook files to visualise the data (with dimensionality reduction) and to perform two-stages clustering. There is one for each dataset (MNIST, Fashion, Cifar).

In the DCEC folder you can find the files to use the DCEC deep clustering algorithm: DCEC.ipynb is the main script. From there, some functions from Varios.py (graphs, metrics, autoencoders etc.) are imported. Additionally, the algorithm is imported from DCEC.py and the data from Datasets. In the folder 'networks' the neural networks already trained are stored.

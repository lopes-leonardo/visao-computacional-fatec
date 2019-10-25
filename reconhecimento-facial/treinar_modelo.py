# COMO USAR
# python treinar_modelo.py
# Ele espera que você já tenha rodado o arquivo extrai_features.py
# Você pode modificar os caminhos de output e input
# por meio dos parâmetros opcionais abaixo

# Importa o pacotes necessários
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("--embeddings", required=False,
	default="output/embeddings.pickle",
	help="Caminho para os embeddings serializados")
ap.add_argument("--recognizer", required=False,
	default="output/recognizer.pickle",
	help="Caminho para o output do treinamento")
ap.add_argument("--le", required=False,
	default="output/le.pickle",
	help="Caminho para o output das labels")
args = vars(ap.parse_args())

# Carrega os embeddings das faces
print("Carregando embedding das faces...")
data = pickle.loads(open(args["embeddings"], "rb").read())

# Codificando labels
print("Codificando labels...")
le = LabelEncoder()
labels = le.fit_transform(data["nomes"])

# Treina o SVM com os embeddings e produz o identificador
print("Treinando modelo...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

# Grava o modelo treinado no disco
f = open(args["recognizer"], "wb")
f.write(pickle.dumps(recognizer))
f.close()

# Grava as labels codificadas no disco
f = open(args["le"], "wb")
f.write(pickle.dumps(le))
f.close()
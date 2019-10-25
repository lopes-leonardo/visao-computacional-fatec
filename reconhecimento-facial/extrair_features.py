from imutils import paths
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=False,
	help="Caminho para o diretório com os rostos para treinamento",
	default="dataset")
ap.add_argument("--embeddings", required=False,
	help="Caminho para o output de embedding das faces",
	default="output/embeddings.pickle")
ap.add_argument("--detector", required=False,
	help="Caminho para o modelo treinado do OpenCV para detecção de faces",
	default="modelo_deteccao_face")
ap.add_argument("--embedding-model", required=False,
	help="Caminho para a rede treinada de extração de features",
	default="openface_nn4.small2.v1.t7")
ap.add_argument("--confidence", type=float, default=0.5,
	help="Nível de confiança mínimo para exitar falsos positivos")
args = vars(ap.parse_args())

# Carrega a rede pre-treinada do Caffe
print("Carregando detector de faces...")
caminhoPrototipo = os.path.sep.join([args["detector"], "deploy.prototxt"])
caminhoModelo = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(caminhoPrototipo, caminhoModelo)
 
# Carrega a facenet para extrair as features dos rostos
print("Carregando extrator...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

# Obtém as imagens da pasta de dataset
print("Buscando imagens...")
caminhosImagem = list(paths.list_images(args["dataset"]))

 
# Inicia a lista de imagens conhecidas e seus nomes
embeddingConhecidos = []
nomesConhecidos = []
 
# initialize the total number of faces processed
total = 0

if len(caminhosImagem) == 0:
	print("O dataset informado está vazio.")
	exit()

# Itera sobre as imagens encontradas
for (i, caminhoImagem) in enumerate(caminhosImagem):
	# Extrai o nome da pessoa do caminho da imagem
	# Esse nome também será a label
	print("processando imagem {}/{}".format(i + 1,
		len(caminhosImagem)))
	nome = caminhoImagem.split(os.path.sep)[-2]

	# Carrega a image, faz um resize para 600x600
	# Depois obtém as novas dimensões
	imagem = cv2.imread(caminhoImagem)
	imagem = imutils.resize(imagem, width=600)
	(h, w) = imagem.shape[:2]

	# Constroi um blol da imagem
	imagemBlob = cv2.dnn.blobFromImage(
		cv2.resize(imagem, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)

	# Aplica o detector na imagem para encontrar os rostos
	detector.setInput(imagemBlob)
	deteccoes = detector.forward()

	# Garante que ao menos um rosto foi encontrado
	if len(deteccoes) > 0:
		# Como fizemos somente imagens com 1 rosto.
		# Utilizamos somente o rosto com maior acerto
		i = np.argmax(deteccoes[0, 0, :, 2])
		confianca = deteccoes[0, 0, i, 2]

		# Garante que o rosto encontrado supera a confiança inserida
		if confianca > args["confidence"]:
			# Recupera a box que enquadra o rosto na imagem
			box = deteccoes[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# Extrai a região de interesse da imagem
			face = imagem[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]

			# Garante que a face tenha um tamanho mínimo necessário
			if fW < 20 or fH < 20:
				continue

			# Controi um blob da região de interesse e
			# Passa esse blob pelo extrator de features
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vetor = embedder.forward()

			# add the nome of the person + corresponding face
			# embedding to their respective lists
			nomesConhecidos.append(nome)
			embeddingConhecidos.append(vetor.flatten())
			total += 1

# Grava os embeddings e os nomes no disco
print("Serializando {} embeddings...".format(total))
data = {"embeddings": embeddingConhecidos, "nomes": nomesConhecidos}
f = open(args["embeddings"], "wb")
f.write(pickle.dumps(data))
f.close()


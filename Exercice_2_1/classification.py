from skimage import io
import numpy as np

# Lire l'image
classification = io.imread("IRC_classif.tif")
segmentation = io.imread("IRC_Segmentation.tif")

shape = classification.shape

classes = np.unique(classification)
segments = np.unique(segmentation)

nbc = len(classes) # nbc = 5
nbs = len(segments) # nbs = 668

M = np.zeros((nbc, nbs))

# Parcourir conjointement les images de segmentation et de classification
for ligne in range(shape[0]):
    for colonne in range(shape[1]):
        classe = classification[ligne, colonne]
        segment = segmentation[ligne, colonne]
       
        # Récupérer l'index du segment dans la liste des segments, pour la classe pas vraiment utile
        id_classe = np.where(classes == classe)[0][0]
        id_segment = np.where(segments == segment)[0][0]
         # Incrémenter l'élément (id_classe, id_segment) de M
        M[id_classe, id_segment] += 1
       
       
# Initialiser un vecteur colonne V de taille (nbs)      
V = np.zeros((nbs, 1))

# Pour chaque segment
for i in range(nbs):
    # Récupérer l'identifiant de la classe max pour ce segment
    id_classe_max = np.argmax(M[:, i])
    # Stocker cet identifiant dans le vecteur V
    V[i] = id_classe_max
           

# Initialiser une image de sortie de taille identique à celle de segmentation
output = np.zeros_like(segmentation)

# Parcourir les pixels de l'image de segmentation
for ligne in range(shape[0]):
    for colonne in range(shape[1]):
        # Récupérer l'identifiant du segment pour ce pixel
        segment = segmentation[ligne, colonne]
        id_segment = np.where(segments == segment)[0][0]
        # Récupérer la valeur correspondant à ce segment dans le vecteur V
        classe_majoritaire = V[id_segment]
        # Affecter cette valeur dans l'image de sortie
        output[ligne, colonne] = classe_majoritaire

# Enregistrer l'image de sortie
io.imsave('classif.tif', output)

# Afficher l'image
# io.imshow(classification)
# io.imshow(segmentation)
io.imshow(output)
io.show()

from skimage import io
import numpy as np

# Lire l'image
classification = io.imread("KoLanta_classification.tif")
segmentation = io.imread("KoLanta_segmentation.tif")

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
        
        
# Afficher l'image
io.imshow(classification)
io.imshow(segmentation)
# io.imshow(output)
io.show()
import numpy as np
from skimage import io

def semantic_improvement(segmentation, classification):
    # Récupérer le nombre de classes et le nombre de segments
    nbc = int(np.max(classification)) + 1
    nbs = int(np.max(segmentation)) + 1
    
    # Créer une matrice M pour stocker le nombre de pixels par classe pour chaque segment
    M = np.zeros((nbc, nbs), dtype=np.int32)
   
    for i in range(nbs):
        mask = segmentation == i
        hist = np.histogram(classification[mask], bins=nbc, range=(0, nbc))[0]
        M[:, i] = hist
        print(hist)
        
    # Créer une image vide pour la nouvelle classification
    new_classification = np.zeros_like(classification)
    
    # Pour chaque segment, déterminer la classe la plus fréquente et affecter tous les pixels de ce segment à cette classe
    for i in range(nbs):
        class_frequencies = M[:, i]
        most_frequent_class = np.argmax(class_frequencies)
        mask = segmentation == i
        new_classification[mask] = most_frequent_class
    
    return new_classification
    
segmentation = io.imread("KoLanta_segmentation.tif")
classification = io.imread("KoLanta_classification.tif")
new_classification = semantic_improvement(segmentation, classification)

# io.imsave("new_classification.tif", new_classification)
io.imshow(new_classification)
io.show()
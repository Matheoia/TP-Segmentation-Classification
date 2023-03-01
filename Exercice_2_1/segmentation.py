from skimage import io
import numpy as np

# Charger l'image IRC et l'image de segmentation
irc_image = io.imread("ImageIRC.tif")
segmentation_image = io.imread("IRC_Segmentation.tif")

# Initialiser l'image de sortie
mean_irc_image = np.zeros_like(irc_image)

# Obtenir les valeurs unique de segmentation_image
unique_segments = np.unique(segmentation_image)

# Pour chaque segment
for segment in unique_segments:
    # Extraire les pixels appartenant au segment
    segment_pixels = irc_image[np.where(segmentation_image == segment)]
    
    # Calculer la moyenne pour chaque canal
    mean_r = np.mean(segment_pixels[:, 0])
    mean_g = np.mean(segment_pixels[:, 1])
    mean_b = np.mean(segment_pixels[:, 2])
    
    # Réaffecter les valeurs de chaque canal dans l'image de sortie
    mean_irc_image[np.where(segmentation_image == segment)] = [mean_r, mean_g, mean_b]

# Enregistrer l'image de sortie
# io.imsave('mean_irc_image.tif', mean_irc_image)

# # Afficher l'image
# io.imshow(irc_image)
io.imshow(mean_irc_image)
io.show()
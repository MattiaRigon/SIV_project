import numpy as np
import cv2
from player import Player

def remove_outliers(matrix, threshold=3):
    """
    Removes outliers from a matrix based on a given threshold.

    Parameters:
    matrix (numpy.ndarray): The input matrix.
    threshold (float): The number of standard deviations away from the mean to consider as an outlier. Default is 3.

    Returns:
    numpy.ndarray: The cleaned matrix with outliers removed.
    """
    mean = np.mean(matrix)
    std = np.std(matrix)
    cleaned_matrix = np.clip(matrix, mean - threshold * std, mean + threshold * std)
    return cleaned_matrix


class SoccerField():

    """
    Represents a soccer field and provides methods for updating and generating heatmaps.

    Attributes:
        image (numpy.ndarray): The image of the soccer field.
        x (int): The width of the soccer field.
        y (int): The height of the soccer field.
        field (numpy.ndarray): The matrix representing the soccer field, one for each squad.

    Methods:
        __init__(self, image): Initializes a SoccerField object.
        update(self, player): Updates the soccer field based on the player's position.
        get_maps(self): Returns the individual heatmaps for each squad.
        generate_heatmaps(self): Generates heatmaps based on the field matrix.
    """
    

    # defines the dimension of the soccer field and create a matrix for each squad x * y
    def __init__(self,image) -> None:
        self.image = image
        self.x = image.shape[1]
        self.y = image.shape[0]
        self.field = np.zeros((self.y,self.x,2))
        pass

    def update(self,player:Player):
        x,y = int(player.soccer_field_position[1]),int(player.soccer_field_position[0])
        squad = player.squad[0]-1
        for i in range(-10,10):
            for j in range(-10,10):
                if i == 0 and j == 0:
                    self.field[x,y,squad] = self.field[x,y,squad] + 100
                elif x+i >= 0 and x+i < self.y and y+j >= 0 and y+j < self.x :
                    self.field[x+i,y+j,squad] = self.field[x+i,y+j,squad] + 100/(abs(i/2)+abs(j/2))
        self.field[x,y,squad] = self.field[x,y,squad] + 100
        pass

    def get_maps(self):
        return self.field[:,:,0],self.field[:,:,1]
    
    def generate_heatmaps(self):

        heatmap1 = self.image.copy()
        heatmap2 = self.image.copy()
        total_heatmap = self.image.copy()

        heatmaps = [(heatmap1,self.field[:,:,0]),(heatmap2,self.field[:,:,1]),(total_heatmap,self.field[:,:,0]+self.field[:,:,1])]

        i = 0

        for heatmap_image,matrix in heatmaps:

            i=i+1

            matrix = remove_outliers(matrix)

            # Normalize the matrix values to [0, 255]
            normalized_matrix = cv2.normalize(matrix, None, 0, 255, cv2.NORM_MINMAX)
            heatmap_color_map = cv2.COLORMAP_JET
            # Apply the color map to the normalized matrix
            heatmap = cv2.applyColorMap(normalized_matrix.astype(np.uint8), heatmap_color_map)
            # Resize heatmap to match the original image size
            heatmap_resized = cv2.resize(heatmap, (heatmap_image.shape[1], heatmap_image.shape[0]))
            # Combine the original image and the heatmap
            result = cv2.addWeighted(heatmap_image, 0.7, heatmap_resized, 0.3, 0)
            cv2.imwrite(f"heatmap{i}.jpg", result)

from soccer_field import SoccerField
import numpy as np
import cv2

def populate_soccer_field(soccer_field:SoccerField, transformed_img,soccer_players, soccer_field_offset,isLeft:bool):
    """
    Populates the soccer field image with the positions of the soccer players.

    Parameters:
    soccer_field (SoccerField): The soccer field object.
    transformed_img (numpy.ndarray): The transformed image.
    soccer_players (list): List of soccer player objects.
    soccer_field_offset (dict): Dictionary containing the offset values for the soccer field.
    isLeft (bool): Flag indicating whether the players are on the left side of the field.

    Returns:
    numpy.ndarray: The soccer field image with the player positions marked.
    """
    soccer_field_copy = soccer_field.image.copy()

    for player in soccer_players:
        x_input, y_input = player.transformed_position

        if isLeft:
            x_output = x_input/(transformed_img.shape[1]) * (soccer_field.image.shape[1]/2 - soccer_field_offset["x"]) + soccer_field_offset["x"]
            y_output = y_input/transformed_img.shape[0] * (soccer_field.image.shape[0] - 2*soccer_field_offset["y"]) + soccer_field_offset["y"]
        else:
            x_output = x_input/(transformed_img.shape[1]) * (soccer_field.image.shape[1]/2 - soccer_field_offset["x"]) + soccer_field.image.shape[1]/2
            y_output = y_input/transformed_img.shape[0] * (soccer_field.image.shape[0] - 2*soccer_field_offset["y"]) + soccer_field_offset["y"]

        player.soccer_field_position = np.array([x_output, y_output])
        center_coordinates = tuple(map(int, player.soccer_field_position))
        center_coordinates_2 = tuple(map(int, player.transformed_position))

        if player.squad == 2:
            cv2.circle(soccer_field_copy, center_coordinates, 5, (255, 0, 0), -1)
            cv2.circle(transformed_img, center_coordinates_2, 5, (0, 0, 0), -1)
        elif player.squad == 1:
            cv2.circle(soccer_field_copy, center_coordinates, 5, (0, 0, 255), -1)
            cv2.circle(transformed_img, center_coordinates_2, 5, (0, 0, 0), -1)

        soccer_field.update(player)


    return soccer_field_copy
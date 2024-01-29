import utils
import numpy as np
import cv2

def populate_soccer_field(soccer_field, transformed_img,soccer_players, soccer_field_offset):

    h, w = transformed_img.shape[:2]
    soccer_field_copy = soccer_field.copy()

    # starting_points = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    # for point in starting_points:
    #     x_input, y_input = point
    #     x_output = x_input/(transformed_img.shape[1]) * (soccer_field.shape[1]/2 - soccer_field_offset["x"]) + soccer_field_offset["x"]
    #     y_output = (y_input/transformed_img.shape[0] * (soccer_field.shape[0] - 2*soccer_field_offset["y"]) + soccer_field_offset["y"])
    #     # point_input = np.array([x_input, y_input, 1])
    #     # point_output_homogeneous = np.dot(transform_matrix, point_input)
    #     # point_output = point_output_homogeneous[:2] / point_output_homogeneous[2]
    #     # x_output, y_output = point_output
    #     center_coordinates = tuple(map(int, np.array([x_output, y_output])))
    #     cv2.circle(soccer_field_copy, center_coordinates, 5, (0, 0, 0), -1)

    # final_points = np.float32(soccer_field_points)
    # transform_matrix = cv2.getPerspectiveTransform(starting_points, final_points)
    for player in soccer_players:
        x_input, y_input = player.transformed_position
        x_output = x_input/(transformed_img.shape[1]) * (soccer_field.shape[1]/2 - soccer_field_offset["x"]) + soccer_field_offset["x"]
        y_output = y_input/transformed_img.shape[0] * (soccer_field.shape[0] - 2*soccer_field_offset["y"]) + soccer_field_offset["y"]
        # point_input = np.array([x_input, y_input, 1])
        # point_output_homogeneous = np.dot(transform_matrix, point_input)
        # point_output = point_output_homogeneous[:2] / point_output_homogeneous[2]
        # x_output, y_output = point_output
        player.soccer_field_position = np.array([x_output, y_output])
        center_coordinates = tuple(map(int, player.soccer_field_position))
        center_coordinates_2 = tuple(map(int, player.transformed_position))

        
        
        if player.squad == 2:
            cv2.circle(soccer_field_copy, center_coordinates, 5, (255, 0, 0), -1)
            cv2.circle(transformed_img, center_coordinates_2, 5, (0, 0, 0), -1)
        elif player.squad == 1:
            cv2.circle(soccer_field_copy, center_coordinates, 5, (0, 0, 255), -1)
            cv2.circle(transformed_img, center_coordinates_2, 5, (0, 0, 0), -1)


    return soccer_field_copy
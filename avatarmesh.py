import math
import random
import sys

import numpy as np
import cv2
import mediapipe as mp
from math import *
import pygame

mp_drawing = mp.solutions.drawing_utils
map_face_mesh = mp.solutions.face_mesh
cap = cv2.VideoCapture(0)
display = pygame.display.set_mode((cap.get(3), cap.get(4)))
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
width = int(cap.get(3))
height = int(cap.get(4))

pygame.display.set_caption("Avatar")

WHITE = (255, 255, 255)
RED = (255, 0, 0)


def convert_cv2_img(img):
    return pygame.image.frombuffer(img.tobytes(), img.shape[1::-1], "RGB")


run = True

projection_matrix = [[1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 0]]


def connect_points(i, j, points, colour):
    pygame.draw.line(display, colour, points[i], points[j], 2)


num_points = 10
cube_points = [n for n in range(num_points)]
cube_points[0] = [[1], [0], [-1.5]]  # backtop left vertice
cube_points[1] = [[1], [0], [0]]  # backtop vertice
cube_points[2] = [[1], [0], [1.5]]  # back  top right vertice

cube_points[3] = [[0], [1.5], [0.6]]  # middle right corner (higher)
cube_points[4] = [[0], [-1.5], [0.5]]  # middle right corner (lower)

cube_points[5] = [[1], [0], [1.5]]  # bottom right vertice
cube_points[6] = [[1], [0], [0]]  # bottom vertice
cube_points[7] = [[1], [0], [-1.5]]  # bottom left vertice

cube_points[8] = [[0], [-1.5], [-0.5]]  # middle left corner (lower)
cube_points[9] = [[0], [1.5], [-0.6]]  # middle left corner (higher)

# Left eyes indices
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

LIPS=[ 61, 146, 91, 181, 84, 17, 314, 405, 321, 375,291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78 ]

def euclaidean_distance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((y - y1) ** 2 + (x - x1) ** 2)
    return distance


def blinkRatio(img, landmarks, right_indices, left_indices):
    # right eyes
    # horizontal lines
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]
    pygame.draw.line(display, (0, 255, 0), rh_right, rh_left, 2)

    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]

    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]
    rhDistance = euclaidean_distance(rh_right, rh_left)
    rvDistance = euclaidean_distance(rv_top, rv_bottom)
    lvDistance = euclaidean_distance(lv_top, lv_bottom)
    lhDistance = euclaidean_distance(lh_right, lh_left)

    if rvDistance != 0:
        reRatio = rhDistance / rvDistance
    else:
        reRatio = 0
    if lvDistance != 0:
        leRatio = lhDistance / lvDistance
    else:
        leRatio = 0

    ratio = (reRatio + leRatio) / 2
    return ratio


def landmarksDetection(img, results, draw=False):
    img_height, img_width = img.shape[:2]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in
                  results.multi_face_landmarks[0].landmark]
    return mesh_coord


# cube_points[6] = [[], [], []]
BLACK = (0, 0, 0)
clock = pygame.time.Clock()
scale = 100
angle_x = angle_y = angle_z = 0
offsetx, offsety = -40, 0
FPS = 360
particles = []
movement_scale = 2


def smooth_angle(old_angle, new_angle, smoothing_factor=0.3):
    return old_angle * (1 - smoothing_factor) + new_angle * smoothing_factor

screen_width, screen_height = display.get_width(), display.get_height()
# Assuming you have previous angles stored
prev_rotx, prev_roty, prev_rotz = 0, 0, 0

with map_face_mesh.FaceMesh(
        static_image_mode=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while run:
        ret, frame = cap.read()
        clock.tick(FPS)
        mx, my = pygame.mouse.get_pos()

        bg = pygame.Surface((height, width))
        bg.fill((43, 18, 33))
        flipped_frame = cv2.flip(frame, 1)
        #  create a copy of the surface
        view = pygame.surfarray.array3d(bg)

        blackbg = view.transpose([1, 0, 2])
        rgb_frame = cv2.cvtColor(flipped_frame, cv2.COLOR_RGB2BGR)
        results = face_mesh.process(rgb_frame)

        for event in pygame.event.get():
            pygame.event.pump()
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                #  convert from rgb to bgr
        blackbg = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
        #  convert from (width, height, channel) to (height, width, channel)

        #  convert from rgb to bgr



        if results.multi_face_landmarks:
            mesh_coords = landmarksDetection(flipped_frame, results)

            mesh_coords = landmarksDetection(flipped_frame, results, False)
            ratio = blinkRatio(flipped_frame, mesh_coords, RIGHT_EYE, LEFT_EYE)

            if ratio > 5.3:
                print("Blinking")

            # cube rots

            image = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB)

            left_eye_p = map_face_mesh.FACEMESH_LEFT_EYE

            for face_landmarks in results.multi_face_landmarks:
                # draws my face
                mp_drawing.draw_landmarks(
                    image=blackbg,
                    landmark_list=face_landmarks,
                    connections=map_face_mesh.FACEMESH_FACE_OVAL,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)



                rotx = (atan2(face_landmarks.landmark[0].y, face_landmarks.landmark[0].z)) * movement_scale
                roty = atan2(face_landmarks.landmark[0].x * cos(rotx), face_landmarks.landmark[0].y) * movement_scale
                rotz = atan2(cos(rotx), sin(rotx) * sin(roty)) * movement_scale
                # Assuming you have previous angles stored

                if rotx != roty != rotz != 0:

                    angle_y = smooth_angle(prev_roty, roty)
                    angle_x = smooth_angle(prev_rotx, rotx)
                    angle_z = smooth_angle(prev_rotz, rotz)
                else:
                    angle_y = roty
                    angle_x = rotx
                    angle_z = rotz
                prev_rotx, prev_roty, prev_rotz = angle_x, angle_y, angle_z

            # Rotation matrices

            rotation_x = [[1, 0, 0],
                          [0, cos(angle_x), -sin(angle_x)],
                          [0, sin(angle_x), cos(angle_x)]]

            rotation_y = [[cos(angle_y), 0, sin(angle_y)],
                          [0, 1, 0],
                          [-sin(angle_y), 0, cos(angle_y)]]

            rotation_z = [[cos(angle_z), -sin(angle_z), 0],
                          [sin(angle_z), cos(angle_z), 0],
                          [0, 0, 1]]
            nose_landmark = face_landmarks.landmark[1]  # Assuming landmark 1 is the nose
            avatar_x = nose_landmark.x * screen_width
            avatar_y = nose_landmark.y * screen_height
            # Combine rotations in the correct order
            rotation_matrix = np.dot(rotation_z, np.dot(rotation_y, rotation_x))
            # converting 3d coords into 2d space
            # x' = (x-z)/ sqrt(2)

            # y' = (x + 2y + z) / sqrt(6)

            points = [0 for _ in range(len(cube_points))]
            i = 0

            # preforms rots

            maxnum = 0
            for point in cube_points:
                rotated_point = np.dot(rotation_matrix, point)
                point_2d = np.dot(projection_matrix, rotated_point)


                x = point_2d[0][0] * scale + avatar_x

                y = point_2d[1][0] * scale + avatar_y

                points[i] = (x + offsetx, y + offsety)

                i += 1

        #   pygame.draw.circle(display, (255, 0, 0), (x + offsetx, y + offsety), 2)
        display.blit(convert_cv2_img(blackbg), (0, 0))

        # draws a cube
        for particle in particles:
            pygame.draw.ellipse(display, BLACK, particle)
            pygame.draw.ellipse(display, RED, particle, 1)
            particle.y += 9
            if particle.y >= display.get_height() + 30:
                particles.pop(particles.index(particle))

        pygame.draw.polygon(display, BLACK, points)
        pygame.draw.polygon(display, BLACK, [points[0], points[2], points[3], points[9], ])  # creates back of mesh
        pygame.draw.polygon(display, BLACK, [points[0], points[9], points[8], points[7]])  # creates a cyan side face
        pygame.draw.polygon(display, (1, 1, 1), [points[0], points[2], points[5], points[7]])  # front face
        pygame.draw.polygon(display, (4, 4, 4), [points[8], points[7], points[5], points[4]])  # draws a top

        pygame.draw.polygon(display, RED, [mesh_coords[index] for index in RIGHT_EYE], 2)
        for j in range(num_points):

            pygame.draw.circle(display, RED, points[j], 3)
            rect = pygame.Rect(points[j][0] - 3, points[j][1] - 3, 6, 6)
            if rect.collidepoint(mx, my):
                print(j)
        #   connect_points()

        if len(particles) < 15:
            size = random.randint(1, 25)
            particles.append(
                pygame.Rect(points[3][0] + random.randint(0, abs(round((points[9][0] - points[3][0]) / 2))),
                            points[3][1] - 30, size, size))
        pygame.display.update()

cap.release()
cv2.destroyAllWindows()
sys.exit()

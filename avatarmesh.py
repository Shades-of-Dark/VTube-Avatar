import math
import sys

import numpy as np
import cv2
import mediapipe as mp
from math import *
import pygame

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
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
    pygame.draw.line(display, colour, points[i], points[j], 5)


cube_points = [n for n in range(6)]
cube_points[0] = [[-1], [0], [-1]]
cube_points[1] = [[-1.5], [-1], [0]]
cube_points[2] = [[-1], [0], [1]]
cube_points[3] = [[-1], [1], [1]]
cube_points[4] = [[-1.5], [2], [0]]
cube_points[5] = [[-1], [1], [-1]]


clock = pygame.time.Clock()
scale = 100
angle_x = angle_y = angle_z = 0
offsetx, offsety = -40, 0
FPS = 360
particles = []
movement_scale = 2
with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        min_detection_confidence=0.5) as face_mesh:
    while run:
        ret, frame = cap.read()
        clock.tick(FPS)
        bg = pygame.Surface((height, width))
        bg.fill((43, 18, 33))
        #  create a copy of the surface
        view = pygame.surfarray.array3d(bg)
        blackbg = view.transpose([1, 0, 2])

        #  convert from rgb to bgr
        blackbg = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
        #  convert from (width, height, channel) to (height, width, channel)

        #  convert from rgb to bgr
        img_bgr = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
        display.fill((42, 28, 53))
        for event in pygame.event.get():
            pygame.event.pump()
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()

        # cube rots

        flipped_frame = cv2.flip(frame, 1)
        results = face_mesh.process(cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB))

        if not results.multi_face_landmarks:
            continue
        image = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB)

        left_eye_p = mp_face_mesh.FACEMESH_LEFT_EYE

        for face_landmarks in results.multi_face_landmarks:
            # draws my face
            mp_drawing.draw_landmarks(
                image=blackbg,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_FACE_OVAL,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

            rotx = (atan2(face_landmarks.landmark[0].y, face_landmarks.landmark[0].z)) * movement_scale
            roty = atan2(face_landmarks.landmark[0].x * cos(rotx), face_landmarks.landmark[0].y) * movement_scale
            rotz = -atan2(cos(rotx), sin(rotx) * sin(roty)) * movement_scale

            right_eye_p = [
                (face_landmarks.landmark[414].x, face_landmarks.landmark[414].y, face_landmarks.landmark[414].z),
                (face_landmarks.landmark[286].x, face_landmarks.landmark[286].y, face_landmarks.landmark[286].z),
                (face_landmarks.landmark[258].x, face_landmarks.landmark[258].y, face_landmarks.landmark[258].z),
                (face_landmarks.landmark[257].x, face_landmarks.landmark[257].y, face_landmarks.landmark[257].z),
                (face_landmarks.landmark[259].x, face_landmarks.landmark[259].y, face_landmarks.landmark[259].z),
                (face_landmarks.landmark[260].x, face_landmarks.landmark[259].y, face_landmarks.landmark[259].z),
                (face_landmarks.landmark[467].x, face_landmarks.landmark[467].y, face_landmarks.landmark[467].z),
                (face_landmarks.landmark[359].x, face_landmarks.landmark[359].y, face_landmarks.landmark[359].z),
                (face_landmarks.landmark[255].x, face_landmarks.landmark[255].y, face_landmarks.landmark[255].z),
                (face_landmarks.landmark[359].x, face_landmarks.landmark[359].y, face_landmarks.landmark[359].z,),
                (face_landmarks.landmark[254].x, face_landmarks.landmark[254].y, face_landmarks.landmark[254].z),
                (face_landmarks.landmark[253].x, face_landmarks.landmark[253].y, face_landmarks.landmark[253].z),
                (face_landmarks.landmark[252].x, face_landmarks.landmark[252].y, face_landmarks.landmark[252].z),
                (face_landmarks.landmark[256].x, face_landmarks.landmark[256].y, face_landmarks.landmark[256].z),
                (face_landmarks.landmark[341].x, face_landmarks.landmark[341].y, face_landmarks.landmark[341].z),
                (face_landmarks.landmark[463].x, face_landmarks.landmark[463].y, face_landmarks.landmark[463].z)]

            angle_y = roty
            angle_x = rotx
            angle_z = rotz

            # a bunch of facial landmarks ig

        rotation_x = [[1, 0, 0],
                      [0, cos(angle_x), -sin(angle_x)],
                      [0, sin(angle_x), cos(angle_x)]]

        rotation_y = [[cos(angle_y), 0, sin(angle_y)],
                      [0, 1, 0],
                      [-sin(angle_y), 0, cos(angle_y)]]

        rotation_z = [[cos(angle_z), -sin(angle_z), 0],
                      [sin(angle_z), cos(angle_z), 0],
                      [0, 0, 1]]

        # converting 3d coords into 2d space
        # x' = (x-z)/ sqrt(2)

        # y' = (x + 2y + z) / sqrt(6)

        points = [0 for _ in range(len(cube_points))]
        right_eye_points = [0 for _ in range(len(right_eye_p))]
        i = 0

        # preforms rots
        display.blit(convert_cv2_img(blackbg), (0, 0))
        maxnum = 0
        for point in cube_points:

            rotate_x = -np.dot(rotation_x, point)
            rotate_y = np.dot(rotation_y, rotate_x)
            rotate_z = np.dot(rotation_z, rotate_y)
            point_2d = np.dot(projection_matrix, rotate_z)

            x = point_2d[0][0] * scale + display.get_width() / 2

            y = point_2d[1][0] * scale + display.get_height() / 2

            points[i] = (x + offsetx, y + offsety)

            i += 1

        #   pygame.draw.circle(display, (255, 0, 0), (x + offsetx, y + offsety), 2)
        c = 0
        for point in right_eye_p:

            rightEyeRotationX = np.dot(rotation_x, point)
            rightEyeRotationY = np.dot(rotation_y, rightEyeRotationX)
            rightEyeRotationZ = np.dot(rotation_z, rightEyeRotationY)
            rightEyePoint2D = np.dot(projection_matrix, rightEyeRotationZ)

            rightEyeX = rightEyePoint2D[0] * 200 + display.get_width() / 2
            rightEyeY = rightEyePoint2D[1] * 200
            right_eye_points[c] = (rightEyeX + offsetx, rightEyeY + offsety)


            c += 1

        # draws a cube

        '''connect_points(0, 1, points, WHITE)
        connect_points(1, 2, points, WHITE)
        connect_points(2, 3, points, WHITE)
        connect_points(3, 4, points, WHITE)
        connect_points(4, 5, points, WHITE)
        connect_points(5, 0, points, WHITE)'''

        pygame.draw.polygon(display, (0, 0, 0), points)

        for i in range(15):
            if i != 14:
                connect_points(i, i + 1, right_eye_points, (0, 0, 255))
            else:
                connect_points(i, 0, right_eye_points, (0, 0, 255))
        pygame.draw.polygon(display, (255, 255, 255), right_eye_points)

        pygame.draw.line(display, (255,255,255), points[3], points[4])
        pygame.draw.circle(display, (255, 0, 0), (points[4][0] -1, points[4][1] - 1), 3)
        pygame.draw.line(display, (255, 255, 255), points[4], points[5])

        pygame.display.update()

    cap.release()
    cv2.destroyAllWindows()
    sys.exit()

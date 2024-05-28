# get image from memory -> image to binary -> find contours ->
# ->  ask for highest point in console -> generate 360 lines from that point to ends -> follow each line, 
#  for each line find intesection height -> plot with PyVista the contours with their height -> fill in mesh


import cv2
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import os
import sys
import random


IMG = 'images\map_big.jpg'

def image_to_contours(img):
    pass

def image_to_binary_sens(img):
    # Convert image to binary more sensitive using adaptive thresholding
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 20)
    return binary

def get_highest_point(contours):
    # Get highest point from console
    print("Enter the highest point: ")
    highest_point = int(input())
    return highest_point

def generate_lines(highest_point):
    # Generate 360 lines from highest point to ends
    lines = []
    for i in range(0, 360):
        lines.append((highest_point, i))
    return lines

def random_dash_pattern():
    return [random.randint(5, 10), random.randint(2, 5)]

def get_image():
    # Get image from memory
    img = cv2.imread(IMG)
    return img

def image_to_binary(img):
    # Convert image to binary
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return binary

def dilate_img(img):
    kernel = np.ones((3,3), np.uint8)  # You can adjust the size of the kernel as needed
    # Perform dilation
    dilated_image = cv2.dilate(img, kernel, iterations=2)
    return dilated_image

def erode_img(img):
    kernel = np.ones((3,3), np.uint8)  # You can adjust the size of the kernel as needed
    # Perform dilation
    eroded_image = cv2.erode(img, kernel, iterations=2)
    return eroded_image

def find_contours(binary):
    # Find contours, but merge very close contours
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return contours

# Remove duplicate contours based on intersection area threshold
def remove_duplicate_contours(contours, threshold=0.05):
    # Indices of contours to be removed
    remove_indices = []
    white_img = cv2.imread('images\white_img.jpg')
    height, width = white_img.shape[:2]

    # Compare each pair of contours
    for i in range(len(contours)):
        for j in range(i + 1, len(contours)):
            intersection_area_i = cv2.contourArea(cv2.convexHull(contours[i]))
            intersection_area_j = cv2.contourArea(cv2.convexHull(contours[j]))
            contour1 = contours[i].astype(np.int32)
            contour2 = contours[j].astype(np.int32)

            # Find convex hulls of the contours
            hull1 = cv2.convexHull(contour1)
            hull2 = cv2.convexHull(contour2)
            black_image1,black_image2 = np.zeros((height, width), dtype=np.uint8),np.zeros((height, width), dtype=np.uint8)

            cv2.drawContours(black_image1, [hull1], -1, 255, thickness=cv2.FILLED)
            cv2.drawContours(black_image2, [hull2], -1, 255, thickness=cv2.FILLED)
            

            # Find the intersection of the binary masks
            intersection = cv2.bitwise_and(black_image1, black_image2)

            # Calculate area of intersection
            intersection_area = np.count_nonzero(intersection)
            # Check if intersection area exceeds threshold for both contours
            if intersection_area < (1+threshold)*intersection_area_j and intersection_area > (1-threshold)*intersection_area_i:
                # Add contour indices to remove list
                #print(f"found {i}, {j}",end=' | ')
                remove_indices.append(i)

    print(remove_indices)
    # Remove duplicate contours
    unique_contours = [contours[i] for i in range(len(contours)) if i not in remove_indices]

    return unique_contours

def main():
    img = get_image()
    binary = image_to_binary_sens(img)
    dilated = dilate_img(binary)
    eroded = erode_img(dilated)

    contours = find_contours(eroded)
    # plot every contour using cv
    print(len(contours))
    contours_after = remove_duplicate_contours(contours)
    print(len(contours_after))


    white_img = cv2.imread('images\white_img.jpg')
    for (i,contour) in enumerate(contours):
        cv2.drawContours(white_img, contour, -1, tuple(random.randint(0, 255) for _ in range(3)), 3)
    cv2.imshow('Contours', white_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()

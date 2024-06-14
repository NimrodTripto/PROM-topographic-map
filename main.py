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
from algorithmic import algorithmic
import algorithmic_advancements
import math
import statistics
import argparse

DEBUG = False
IMG1 = 'images\map_small.png'
IMG1 = 'images\map_big.jpg'
IMG = 'images\map_hand2.jpg'
IMG1 = 'images\map2.jpg'

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

def get_image(image=IMG):
    # Get image from memory
    img = cv2.imread(image)
    return img

def image_to_binary(img):
    # Convert image to binary
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return binary

def dilate_img(img,shape=(2,2),iter=2):
    kernel = np.ones(shape, np.uint8)  # You can adjust the size of the kernel as needed
    # Perform dilation
    dilated_image = cv2.dilate(img, kernel, iterations=iter)
    return dilated_image

def erode_img(img,shape=(2,2),iter=2):
    kernel = np.ones(shape, np.uint8)  # You can adjust the size of the kernel as needed
    # Perform dilation
    eroded_image = cv2.erode(img, kernel, iterations=iter)
    return eroded_image

def close_img(img,shape=(2,2),iter=2):
    kernel = np.ones(shape, np.uint8)  # You can adjust the size of the kernel as needed
    # Perform dilation
    closed_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=iter)
    return closed_img

def open_img(img,shape=(2,2),iter=2):
    kernel = np.ones(shape, np.uint8)  # You can adjust the size of the kernel as needed
    # Perform dilation
    opened_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=iter)
    return opened_img


def find_contours(binary):
    # Find contours, but merge very close contours
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return contours

def intersection_test1(contour1,contour2,height, width,debug = False, threshold=0.1):
    intersection_area_i = cv2.contourArea(cv2.convexHull(contour1))
    intersection_area_j = cv2.contourArea(cv2.convexHull(contour2))
    contour1 = contour1.astype(np.int32)
    contour2 = contour2.astype(np.int32)

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
    if(debug):
        print(intersection_area_i,intersection_area_j,intersection_area)
        cv2.imshow('Contours', black_image1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imshow('Contours', intersection)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    # Check if intersection area exceeds threshold for both contours
    if intersection_area < (1+threshold)*intersection_area_j and intersection_area < (1+threshold)*intersection_area_i:
        if intersection_area > (1-threshold)*intersection_area_j and intersection_area > (1-threshold)*intersection_area_i:
            return True
    return False

def try_sample(contour,num_sample):
    if(len(contour)>=num_sample):
        contour_points = random.sample(range(len(contour)), num_sample)
    else:
        contour_points = random.sample(range(len(contour)), len(contour))
    return contour_points

def intersection_test2(contour1,contour2, threshold=10):
    # all_contour1_points = [pt[0] for pt in contour1]
    # contour1_points = try_sample(contour1,100)
    contour2_points = try_sample(contour2,10)
    all_dists = []
    for pt_i in contour2_points:
        curr_dist = min(math.dist(contour1[pt_j][0],contour2[pt_i][0]) for pt_j in range(len(contour1)))
        all_dists.append(curr_dist)
        if(curr_dist > 4 * threshold):
            return False
    return statistics.mean(all_dists)<threshold



# Remove duplicate contours based on intersection area threshold
def remove_duplicate_contours(contours, threshold=10):
    # Indices of contours to be removed
    remove_indices = []
    white_img = cv2.imread('images\white_img.jpg')
    height, width = white_img.shape[:2]

    # Compare each pair of contours
    for i in range(len(contours)):
        for j in range(i + 1, len(contours)):
            # if (intersection_test1(contours[i],contours[j],height, width) or intersection_test2(contours[i],contours[j])):
            # if (intersection_test1(contours[i],contours[j],height, width)):
            #         remove_indices.append(i)
            # elif (intersection_test2(contours[i],contours[j])):
            #     remove_indices.append(i)
            if (intersection_test2(contours[i],contours[j], threshold)):
                remove_indices.append(i)
                

    # print(remove_indices)
    # Remove duplicate contours
    unique_contours = [contours[i] for i in range(len(contours)) if i not in remove_indices]

    return unique_contours

# Remove duplicate contours based on intersection area threshold
def remove_small_contours(contours, threshold=10):
    # Indices of contours to be removed
    remove_indices = []
    white_img = cv2.imread('images\white_img.jpg')
    height, width = white_img.shape[:2]

    # Compare each pair of contours
    for i in range(len(contours)):
        contour_area = cv2.contourArea(cv2.convexHull(contours[i]))
        if (contour_area<threshold):
                remove_indices.append(i)
                

    # print(remove_indices)
    # Remove duplicate contours
    unique_contours = [contours[i] for i in range(len(contours)) if i not in remove_indices]

    return unique_contours

def main(img_path):
    img = get_image(img_path)
    binary = image_to_binary_sens(img)
    curr_img = binary
    curr_img = dilate_img(curr_img,(2,2),2)
    curr_img = erode_img(curr_img,(2,2),2)
    # cv2.imshow('Contours', curr_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    contours = find_contours(curr_img)
    # plot every contour using cv
    contours_after = remove_duplicate_contours(remove_small_contours(contours,10), 5)
    # contours_after = contours
    #i==4
    white_img = cv2.imread('images\white_img.jpg')
    # if(DEBUG):
    #     for (i,contour) in enumerate(contours_after):
    #         if(i==0 or i==7):
    #             # print(contour)
    #             # print(algorithmic_advancements.contour_to_r_theta(contour,[500,500]))
    #             cv2.drawContours(white_img, contour, -1, tuple(random.randint(0, 255) for _ in range(3)), 3)
    #     cv2.imshow('Contours', white_img)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    # else:
    #     for (i,contour) in enumerate(contours_after):
    #         cv2.drawContours(white_img, contour, -1, tuple(random.randint(0, 255) for _ in range(3)), 3)
        # cv2.imshow('Contours', white_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    contours_after.pop(7)
    contours_after.pop(0)
    # algorithmic([contours[15]], img.shape[:2])
    algorithmic(contours_after, img.shape[:2])



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process an image to find contours.')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    if img is None:
        print(f"Error: Unable to load image from path {args.image_path}")
        sys.exit(1)
    main(args.image_path)
    

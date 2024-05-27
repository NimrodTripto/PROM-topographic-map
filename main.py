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
INITIAL_HGT = 3000


def get_image():
    # Get image from memory
    img = cv2.imread(IMG)
    return img

def image_to_binary(img):
    # Convert image to binary
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return binary

def image_to_contours(img):
    pass

def image_to_binary_sens(img):
    # Convert image to binary more sensitive using adaptive thresholding
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 20)
    return binary

def find_contours(binary):
    # Find contours, but merge very close contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

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

def main():
    img = get_image()
    binary = image_to_binary_sens(img)
    contours = find_contours(binary)

    # show binary image
    cv2.imshow('Binary', binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # plot every contour using cv
    for contour in contours:
        cv2.drawContours(img, contour, -1, (0, 255, 0), 3)
    cv2.imshow('Contours', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def find_smallest_closed_contour(contours):
    # this will find the "peaks" of the map, the smallest closed contours
    # they have the smallest area and they don't contain any other contour
    smallest_contour = None
    smallest_area = 1000000
    smallest_contour_list = []
    for contour in contours:
        if cv2.pointPolygonTest(smallest_contour, tuple(contour[0][0]), False) == -1:
            smallest_contour_list.append(contour)            


def algorithmic():
    # This function gets a list of contours in the "find contours" format, and makes the algorithmic part
    # so that it will be able, in the end, to model the map
    
    # draw map in 3D
    plotter = pv.Plotter()
    plotter.add_mesh(pv.Cube())
    plotter.show()


if __name__ == "__main__":
    main()

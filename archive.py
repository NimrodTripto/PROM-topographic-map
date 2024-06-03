# def is_contour_closed3(contour, X_MAX=1000, Y_MAX=1000):
#     """
#     Check if the contour is closed based on the mean point and the furthest point.

#     Parameters:
#     contour (np.ndarray): The contour to check.
#     X_MAX (float): The maximum x-coordinate value.
#     Y_MAX (float): The maximum y-coordinate value.

#     Returns:
#     bool: True if the contour is closed, False otherwise.
#     """
#     # Ensure contour shape is (N, 1, 2)
#     contour_points = contour.squeeze()  # Shape (N, 2)

#     # Calculate the mean point of the contour
#     mean_point = np.mean(contour_points, axis=0)
#     xm, ym = mean_point

#     # Find the furthest point from the mean point
#     distances = np.linalg.norm(contour_points - mean_point, axis=1)
#     max_index = np.argmax(distances)
#     d = distances[max_index]

#     # Check the conditions for closure
#     if (xm - 0)**2 <= d**2 or (xm - X_MAX)**2 <= d**2:
#         return True
#     if (ym - 0)**2 <= d**2 or (ym - Y_MAX)**2 <= d**2:
#         return True

#     return False

# def is_contour_closed(contour, threshold=THRESH):
#     """
#     Check if the contour is closed.

#     Parameters:
#     contour (np.ndarray): The contour to check.
#     threshold (float): The threshold to consider the contour closed.

#     Returns:
#     bool: True if the contour is closed, False otherwise.
#     """
#     # Calculate the arc length of the contour assuming it's closed
#     closed_arc_length = cv2.arcLength(contour, True)
    
#     # Calculate the arc length of the contour assuming it's open
#     open_arc_length = cv2.arcLength(contour, False)
    
#     # The contour is closed if the arc lengths are approximately equal
#     return abs(closed_arc_length - open_arc_length) < threshold

# def plot_contour_and_points(contour, points, results):
#     """
#     Plot the contour and the points being tested.

#     Parameters:
#     contour (np.ndarray): The contour to plot.
#     points (list of tuples): The points being tested.
#     results (list of float): The results of pointPolygonTest for each point.
#     """
#     plt.figure()
#     contour_points = contour.squeeze()
#     plt.plot(contour_points[:, 0], contour_points[:, 1], 'b-', label='Contour')
#     for point, result in zip(points, results):
#         color = 'r' if result == 1.0 else 'g'
#         plt.plot(point[0], point[1], marker='o', color=color)
#     plt.legend()
#     plt.show()


# def count_contained_contours(contour, all_contours):
#         if not is_valid_contour_shape(contour):
#             raise ValueError(f"Contour format is incorrect. Expected shape is (N, 1, 2). Got shape {contour.shape}")

#         count = 0
#         for idx, other_contour in all_contours.items():
#             if other_contour is not contour:
#                 if not is_valid_contour_shape(other_contour):
#                     raise ValueError(f"Other contour format is incorrect. Expected shape is (N, 1, 2). Got shape {other_contour.shape}")
            
#                 point = (int(other_contour[0][0][0]), int(other_contour[0][0][1]))

#                 result = cv2.pointPolygonTest(contour, point, False)

#                 if result == 1.0:
#                     count += 1
#         return count

# def calculate_heights_old(contours, img_shape, initial_height=INITIAL_HGT):
#     contour_with_heights = {}
#     for i, contour in contours.items():
#         if is_contour_closed2(contour, X_MAX=img_shape[1], Y_MAX=img_shape[0]):
#             num_contained_contours = count_contained_contours(contour, contours)
#             height = initial_height - DIFF * num_contained_contours # fix
#             contour_with_heights[i] = (height, contour)
#             print(f"Contour {i} has height {height}")
    
#     return contour_with_heights

# def calculate_heights(contours, img_shape, initial_height=INITIAL_HGT):
#     contour_with_heights = {}
#     assigned_contours = {}
#     # Assign tops with initial height
#     for i, contour in contours.items():
#         if is_contour_closed2(contour, X_MAX=img_shape[1], Y_MAX=img_shape[0]):
#             num_contained_contours = count_contained_contours(contour, contours)
#             if num_contained_contours == 0:
#                 height = initial_height
#                 assigned_contours.add(i)
#             else:
#                 height = -1
#             contour_with_heights[i] = (height, contour)

#     # Assign heights to the rest of the contours
#     while len(assigned_contours) != len{contours}:
#         for i, contour in contours.items():
#             if i in assigned_contours:
#                 continue
#             num_contained_contours = count_contained_contours(contour, contours)
#             if all([j in assigned_contours for j in num_contained_contours]):
#                 height = initial_height - DIFF * num_contained_contours
#                 contour_with_heights[i] = (height, contour)
#                 assigned_contours.add(i)
#                 print(f"Contour {i} has height {height}")
        
    
#     return contour_with_heights

# def plot_3d_model_from_dict(contours_with_heights, diff=DIFF):
#     # Initialize the plotter
#     plotter = pv.Plotter()

#     # Collect all points and faces
#     all_points = []
#     faces = []

#     # Loop through the contours and add them to the mesh with the respective heights
#     for height, contour in contours_with_heights.values():
#         points = contour.squeeze()

#         # Top height points
#         z_values_top = np.full((points.shape[0], 1), height)
#         points_top = np.hstack((points, z_values_top))

#         # Bottom height points
#         z_values_bottom = np.full((points.shape[0], 1), height - diff)
#         points_bottom = np.hstack((points, z_values_bottom))

#         num_points = points_top.shape[0]

#         # Add points to all_points list and keep track of indices
#         top_indices = []
#         bottom_indices = []
#         for pt in points_top:
#             top_indices.append(len(all_points))
#             all_points.append(pt)
#         for pt in points_bottom:
#             bottom_indices.append(len(all_points))
#             all_points.append(pt)

#         # Create faces for the top and bottom surfaces
#         for i in range(num_points):
#             faces.append([4, top_indices[i], top_indices[(i + 1) % num_points], bottom_indices[(i + 1) % num_points], bottom_indices[i]])

#     # Flatten the points and faces list
#     all_points = np.array(all_points)
#     faces = np.hstack(faces)

#     # Create a PolyData object
#     poly = pv.PolyData()
#     poly.points = all_points
#     poly.faces = faces

#     # Add the mesh to the plotter
#     plotter.add_mesh(poly, color='purple', opacity=1.0, show_edges=True)

#     # Show the plot
#     plotter.show()
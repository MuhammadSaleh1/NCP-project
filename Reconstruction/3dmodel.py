import cv2
import open3d as o3d
import numpy as np

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize Open3D visualizer
vis = o3d.visualization.Visualizer()
vis.create_window()

# Initialize point cloud
pcd = o3d.geometry.PointCloud()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect features
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    # Draw keypoints (for visualization purposes)
    frame_with_keypoints = cv2.drawKeypoints(frame, keypoints, None, color=(0, 255, 0))

    # Convert frame to point cloud (dummy example)
    # In practice, you would use depth information here
    height, width = gray.shape
    for y in range(height):
        for x in range(width):
            pcd.points.append([x, y, gray[y, x] / 255.0])

    # Update point cloud in Open3D
    vis.add_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()

    # Display the resulting frame
    cv2.imshow('Frame with Keypoints', frame_with_keypoints)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
vis.destroy_window()

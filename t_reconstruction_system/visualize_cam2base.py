import open3d as o3d
import numpy as np
import argparse

def lineset_from_extrinsics(extrinsics):
    POINTS_PER_FRUSTUM = 5
    EDGES_PER_FRUSTUM = 8

    points = []
    colors = []
    lines = []

    cnt = 0
    for extrinsic in extrinsics:
        # mm to m
        extrinsic[:3, 3] *= 0.001

        # For debug: I'm not sure if it is inverted
        pose = extrinsic
        # pose = np.linalg.inv(extrinsic)

        # Adjust this number to change the size of the frustum
        l = 0.02
        points.append((pose @ np.array([0, 0, 0, 1]).T)[:3])
        points.append((pose @ np.array([l, l, 2 * l, 1]).T)[:3])
        points.append((pose @ np.array([l, -l, 2 * l, 1]).T)[:3])
        points.append((pose @ np.array([-l, -l, 2 * l, 1]).T)[:3])
        points.append((pose @ np.array([-l, l, 2 * l, 1]).T)[:3])

        lines.append([cnt + 0, cnt + 1])
        lines.append([cnt + 0, cnt + 2])
        lines.append([cnt + 0, cnt + 3])
        lines.append([cnt + 0, cnt + 4])
        lines.append([cnt + 1, cnt + 2])
        lines.append([cnt + 2, cnt + 3])
        lines.append([cnt + 3, cnt + 4])
        lines.append([cnt + 4, cnt + 1])

        for i in range(0, EDGES_PER_FRUSTUM):
            colors.append(np.array([1, 0, 0]))

        cnt += POINTS_PER_FRUSTUM

    for i in range(len(extrinsics) - 1):
        s = i
        t = i + 1
        lines.append([POINTS_PER_FRUSTUM * s, POINTS_PER_FRUSTUM * t])
        colors.append(np.array([0, 1, 0]))

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(np.vstack(points))
    lineset.lines = o3d.utility.Vector2iVector(np.vstack(lines).astype(int))
    lineset.colors = o3d.utility.Vector3dVector(np.vstack(colors))

    return lineset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    args = parser.parse_args()

    extrinsics = np.load(args.path)
    lineset = lineset_from_extrinsics(extrinsics)
    o3d.visualization.draw([lineset])

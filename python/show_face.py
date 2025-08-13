import cv2
import numpy as np
import imageio

class FaceCubeVisualizer:
    def __init__(self, img_size=(480,640), cube_size=200, camera_matrix=None):
        self.img_h, self.img_w = img_size
        self.cube_size = cube_size

        if camera_matrix is None:
            self.camera_matrix = np.array([[800,0,self.img_w//2],
                                           [0,800,self.img_h//2],
                                           [0,0,1]], dtype=np.float32)
        else:
            self.camera_matrix = camera_matrix

        self.dist_coeffs = np.zeros(5, dtype=np.float32)
        self.head_center = np.array([0,0,1000], dtype=np.float32)

        s = self.cube_size/2
        self.cube_vertices = np.float32([
            [-s,-s,-s],[ s,-s,-s],[ s, s,-s],[-s, s,-s],  # 背面
            [-s,-s, s],[ s,-s, s],[ s, s, s],[-s, s, s],  # 前面
        ])

        self.face_plane_z = -s
        self.left_eye_face  = np.array([-50, -25, 0], dtype=np.float32)
        self.right_eye_face = np.array([50, -25, 0], dtype=np.float32)
        self.mouth_face     = np.array([0, 40, 0], dtype=np.float32)

    def rotation_matrix_y(self, angle_deg):
        theta = np.deg2rad(angle_deg)
        return np.array([
            [np.cos(theta),0,np.sin(theta)],
            [0,1,0],
            [-np.sin(theta),0,np.cos(theta)]
        ],dtype=np.float32)

    def _project_point(self, point_3d, R):
        rotated = R @ point_3d.reshape(3,1)
        translated = rotated + self.head_center.reshape(3,1)
        projected, _ = cv2.projectPoints(translated.reshape(1,3),
                                         np.zeros((3,1),dtype=np.float32),
                                         np.zeros((3,1),dtype=np.float32),
                                         self.camera_matrix,
                                         self.dist_coeffs)
        return tuple(projected[0].ravel().astype(int))

    def draw_cube_faces(self, img, R):
        faces = {
            'back':   [4,5,6,7],
            'left':   [0,3,7,4],
            'right':  [1,2,6,5],
            'top':    [0,1,5,4],
            'bottom': [3,2,6,7],
            'front':  [0,1,2,3],
        }
        colors = {
            'back':   (40,70,100),
            'left':   (50,90,130),
            'right':  (50,90,130),
            'top':    (60,110,160),
            'bottom': (60,110,160),
            'front':  (80,140,200),
        }

        # 按深度排序：远 -> 近
        face_depths = []
        for name, vert_idx in faces.items():
            pts_3d = np.array([R @ self.cube_vertices[i] + self.head_center for i in vert_idx])
            z_mean = np.mean(pts_3d[:,2])
            face_depths.append((z_mean, name, vert_idx))
        face_depths.sort(reverse=True)  # 远 -> 近

        for _, name, vert_idx in face_depths:
            pts_3d = [self.cube_vertices[i] for i in vert_idx]
            proj_pts = [self._project_point(p, R) for p in pts_3d]
            proj_pts = np.array(proj_pts, dtype=np.int32)
            cv2.fillPoly(img, [proj_pts], color=colors[name])

    def draw_ellipse_on_face(self, img, center_face, width, height, R, color=(0,0,0), thickness=2, num_points=50):
        angles = np.linspace(0,2*np.pi,num_points)
        ellipse_pts = np.stack([width/2*np.cos(angles), height/2*np.sin(angles), np.zeros_like(angles)], axis=1)
        ellipse_pts += center_face
        ellipse_pts[:,2] += self.face_plane_z
        proj_pts = [self._project_point(p, R) for p in ellipse_pts]
        proj_pts = np.array(proj_pts, dtype=np.int32)
        cv2.polylines(img, [proj_pts], isClosed=True, color=color, thickness=thickness)

    def visualize_frame(self, left_eye_ratio=1.0, right_eye_ratio=1.0, mouth_ratio=1.0, R=None):
        frame = np.ones((self.img_h,self.img_w,3),np.uint8)*255
        if R is None:
            R = np.eye(3, dtype=np.float32)
        self.draw_cube_faces(frame, R)
        # 五官
        self.draw_ellipse_on_face(frame, self.left_eye_face, width=50, height=50*0.3*left_eye_ratio, R=R)
        self.draw_ellipse_on_face(frame, self.right_eye_face, width=50, height=50*0.3*right_eye_ratio, R=R)
        self.draw_ellipse_on_face(frame, self.mouth_face, width=70, height=100*0.25*mouth_ratio, R=R)
        return frame

    def generate_head_shake_gif(self, left_eye_ratio=1.0, right_eye_ratio=1.0, mouth_ratio=0.5,
                                 angle_range=45, step=5, save_path="face_cube_headshake.gif"):
        angles_forward = list(range(-angle_range, angle_range+1, step))
        angles_backward = list(range(angle_range-step, -angle_range-1, -step))
        angles = angles_forward + angles_backward

        frames = []
        for angle in angles:
            R = self.rotation_matrix_y(angle)
            frame = self.visualize_frame(left_eye_ratio, right_eye_ratio, mouth_ratio, R)
            frames.append(frame[:,:,::-1])  # BGR -> RGB
        imageio.mimsave(save_path, frames, duration=0.05)

if __name__=="__main__":
    vis = FaceCubeVisualizer()
    # 单帧测试
    R = vis.rotation_matrix_y(0)
    frame = vis.visualize_frame(left_eye_ratio=0.8, right_eye_ratio=0.6, mouth_ratio=0.5, R=R)
    cv2.imwrite("face_cube_single.png", frame)

    # 左右摇头动画
    vis.generate_head_shake_gif(left_eye_ratio=0.9, right_eye_ratio=0.7, mouth_ratio=0.5,
                                angle_range=45, step=5,
                                save_path="face_cube_headshake.gif")

import path_generation as pg
from path_interpolation import SplineTrajectory3D
import numpy as np

class Scene:
    def __init__(self):
        self.target_position = ()
        self.interferer_position = []
        self.recorded_target = None     

def prepare_position(position, t):
    if isinstance(position, tuple):
        pos, t2 = position
        return SplineTrajectory3D(t2, pos).get(t)
    return position

def create_prepared_scene(scene, duration, sr):
    prepared_scene = Scene()
    t = np.linspace(0, duration, duration*sr)
    prepared_scene.target_position = prepare_position(scene.target_position, t)
    prepared_scene.interferer_position = [prepare_position(i, t) for i in scene.interferer_position]
    return prepared_scene

def scene_center(duration, sr):
    scene = Scene()
    scene.target_position = np.array([0, 0, 10.0])
    return create_prepared_scene(scene, duration, sr)

def scene_stationary(duration, sr, position):
    scene = Scene()
    scene.target_position = position
    return create_prepared_scene(scene, duration, sr)

def scene_line(duration, sr, start, stop):
    scene = Scene()
    scene.target_position = pg.straight_line(3, duration, start, stop)
    return create_prepared_scene(scene, duration, sr)

def get_position(position, t_idx):
    if isinstance(position, tuple):
        return position[0][:, t_idx]
    else:
        return position

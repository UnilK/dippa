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

def scene_stationary(duration, sr):
    scene = Scene()
    scene.target_position = np.array([2.0, 5.0, 5.0])
    return create_prepared_scene(scene, duration, sr)

def scene_line(duration, sr):
    scene = Scene()
    scene.target_position = pg.straight_line(
        3,
        duration,
        np.array([5, -5, 5]),
        np.array([5, 5, 5]))
    return create_prepared_scene(scene, duration, sr)

def scene_circle(duration, sr):
    scene = Scene()
    scene.target_position = pg.circular_orbit(
        100,
        duration,
        radius=5)
    return create_prepared_scene(scene, duration, sr)

def scene_circle2(duration, sr):
    scene = Scene()
    scene.target_position = pg.circular_orbit(
        100,
        duration,
        center=np.array([0.0, 0.0, 0.0]),
        radius = 7)
    return create_prepared_scene(scene, duration, sr)

def scene_helix(duration, sr):
    scene = Scene()
    scene.target_position = pg.helix_climb(
        100,
        duration,
        radius=3,
        height=9,
        turns = 2)
    return create_prepared_scene(scene, duration, sr)

def scene_random(duration, sr):
    scene = Scene()
    scene.target_position = pg.random_waypoints_spline(
        6,
        duration,
        np.array([-7, -7, 1]),
        np.array([7, 7, 10])
    )
    return create_prepared_scene(scene, duration, sr)

def scene_zeni(duration, sr):
    scene = Scene()
    t = np.linspace(0.0, duration, 200)
    theta = np.pi * (t / duration)

    x = 7 * np.cos(theta)
    y = np.full_like(x, 0)
    z = 7 * np.sin(theta)

    points = np.vstack((x, y, z))
    scene.target_position = (points, t)
    return create_prepared_scene(scene, duration, sr)

def scene_interferer(duration, sr):
    scene = Scene()
    scene.target_position = pg.circular_orbit(
        100,
        duration,
        radius=5)
    scene.interferer_position.append(np.array([2.0, 5.0, 5.0]))
    scene.interferer_audio.append((np.random.randn(duration*sr), sr))
    return create_prepared_scene(scene, duration, sr)

def get_position(position, t_idx):
    if isinstance(position, tuple):
        return position[0][:, t_idx]
    else:
        return position

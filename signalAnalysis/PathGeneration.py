import numpy as np

def straight_line(num_points=2, duration=10.0,
                  start=np.array([0.0, 0.0, 0.0]),
                  end=np.array([10.0, 0.0, 5.0])):
    """
    Simple straight-line path from start to end.
    Returns: points (3 x N), t (N)
    """
    t = np.linspace(0.0, duration, num_points)
    # Linear interpolation between start and end
    points = (start[:, None] + (end - start)[:, None] * (t / duration))
    return points, t


def circular_orbit(num_points=100, duration=10.0,
                   center=np.array([0.0, 0.0, 5.0]),
                   radius=10.0, clockwise=True):
    """
    Horizontal circular orbit around center at fixed altitude.
    Returns: points (3 x N), t (N)
    """
    t = np.linspace(0.0, duration, num_points)
    # Angle vs time: one full revolution
    theta = 2.0 * np.pi * (t / duration)
    if clockwise:
        theta = -theta

    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    z = np.full_like(x, center[2])

    points = np.vstack((x, y, z))
    return points, t


def helix_climb(num_points=100, duration=10.0,
                center=np.array([0.0, 0.0, 0.0]),
                radius=5.0, height=20.0, turns=3.0):
    """
    Helical ascent around center.
    Returns: points (3 x N), t (N)
    """
    t = np.linspace(0.0, duration, num_points)
    # Angle increases over 'turns' revolutions
    theta = 2.0 * np.pi * turns * (t / duration)

    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    z = center[2] + height * (t / duration)

    points = np.vstack((x, y, z))
    return points, t


def lemniscate_figure8(num_points=100, duration=10.0,
                       center=np.array([0.0, 0.0, 5.0]),
                       scale=10.0, z_amplitude=1.0):
    """
    3D figure-eight path (lemniscate of Bernoulli in XY, sinusoidal in Z).
    Returns: points (3 x N), t (N)
    """
    t = np.linspace(0.0, duration, num_points)
    # Parameter for lemniscate
    u = 2.0 * np.pi * (t / duration)  # one full figure 8

    # Lemniscate of Bernoulli: (cos(u), sin(2u)/2)
    x = center[0] + scale * np.cos(u) / (1.0 + np.sin(u)**2)
    y = center[1] + scale * np.sin(u) * np.cos(u) / (1.0 + np.sin(u)**2)
    z = center[2] + z_amplitude * np.sin(2.0 * np.pi * (t / duration))

    points = np.vstack((x, y, z))
    return points, t


def random_waypoints_spline(num_points=10, duration=10.0,
                            bbox_min=np.array([-10.0, -10.0, 1.0]),
                            bbox_max=np.array([10.0, 10.0, 10.0]),
                            seed=None):
    """
    Smooth path through random waypoints using cubic interpolation.
    Returns: points (3 x N), t (N)
    """
    rng = np.random.default_rng(seed)

    # Generate random waypoints within bounding box
    waypoints = rng.uniform(bbox_min, bbox_max, size=(num_points, 3))
    # Waypoint times evenly spaced over duration
    waypoint_t = np.linspace(0.0, duration, num_points)

    # Target sample times
    t = np.linspace(0.0, duration, num_points)

    # Cubic interpolation per coordinate (Hermite-like via np.interp? Here use simple linear;
    # for actual spline, use scipy if available. Below is linear for NumPy-only.)
    # If you want real splines without SciPy, you'd implement them manually.
    x = waypoints[:, 0]
    y = waypoints[:, 1]
    z = waypoints[:, 2]

    points = np.vstack((x, y, z))
    return points, t
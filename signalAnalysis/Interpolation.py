import numpy as np
from scipy.interpolate import CubicSpline

class SplineTrajectory3D:
    """
    Represents a time-parameterized 3D trajectory using cubic splines.
    Velocity at the start and end is constrained to be zero.
    """

    def __init__(self, t, points):
        t = np.asarray(t, dtype=float)
        points = np.asarray(points, dtype=float)

        if t.ndim != 1:
            raise ValueError("t must be a 1D array of timestamps.")
        if points.ndim != 2 or points.shape[0] != 3:
            raise ValueError("points must be a 2D array of shape (N, 3).")
        if len(t) != points.shape[1]:
            raise ValueError("t and points must have the same length.")
        if not np.all(np.diff(t) > 0):
            raise ValueError("timestamps t must be strictly increasing.")

        self.t = t
        self.points = points

        # Enforce zero velocity at start and end using clamped boundary conditions.
        bc_type = ((1, 0.0), (1, 0.0))

        # Create a separate spline for each coordinate: x(t), y(t), z(t)
        self.splines = []
        for dim in range(3):
            coord = points[dim]
            # Cubic spline that also allows derivative evaluation
            spline = CubicSpline(t, coord, bc_type=bc_type)
            self.splines.append(spline)

        # Convenience bounds
        self.t_min = t[0]
        self.t_max = t[-1]

    def get(self, t_query):
        """
        Get interpolated position and velocity at time t_query.

        Parameters
        ----------
        t_query : float or array_like
            Time(s) at which to evaluate the trajectory.

        Returns
        -------
        pos : ndarray, shape (3,) or (M, 3)
            Interpolated 3D position(s) at t_query.
        vel : ndarray, shape (3,) or (M, 3)
            Interpolated 3D velocity vector(s) at t_query.
        """
        t_query = np.asarray(t_query, dtype=float)

        # Optional: clamp t_query to the valid interval
        t_eval = np.clip(t_query, self.t_min, self.t_max)

        # Evaluate position and first derivative (velocity) for each coordinate
        pos_components = []
        vel_components = []
        for spline in self.splines:
            pos_components.append(spline(t_eval))
            vel_components.append(spline(t_eval, 1))  # derivative wrt time

        pos = np.stack(pos_components)
        vel = np.stack(vel_components)

        return pos, vel
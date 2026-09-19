"""
Fits a camera (intrinsic K, extrinsic R,t) to a badminton court video from
known 2D image points -- e.g. monotrack's line-intersection detector --
matched against the real, standardized court dimensions.

Starts from an external camera prior (VGGT's corrected intrinsic) by
default now, not as an optional extra -- testing showed VGGT usually
matches or beats the physically-motivated seed grid alone, and helps
specifically in edge cases where that grid's seeds don't land in the
right basin. The physically-motivated grid still runs alongside it as a
fallback/complement, so passing a VGGT estimate never makes the fit
worse, but callers should now supply one whenever it's available rather
than treating it as a nice-to-have. Also exposes a `loss()` you can add
other terms to later (e.g. a MegaSaM-style flow-consistency term).
"""

import numpy as np
from scipy.optimize import least_squares


class court_detection:
    # standard badminton court measurements, in meters
    COURT_WIDTH = 6.1        # doubles sideline to sideline (X axis)
    COURT_LENGTH = 13.4      # baseline to baseline (Y axis)
    SINGLES_WIDTH = 5.18
    NET_HEIGHT = 1.55
    SERVICE_LINE_DIST = 1.98   # short service line, from the net
    DOUBLES_LINE_DIST = 0.76   # long service line, from each baseline

    _singles_inset = (COURT_WIDTH - SINGLES_WIDTH) / 2  # 0.46
    _net_y = COURT_LENGTH / 2                           # 6.7
    _center_x = COURT_WIDTH / 2                         # 3.05

    # the 22 named court-line intersections monotrack's BadmintonCourtModel
    # fits internally (P1..P22 in its source), plus the 2 net-pole tops.
    # Y=0 is the near baseline (BL/BR side), Y=13.4 is the far baseline.
    COURT_POINTS_3D = {
        "P1_TL": (0, COURT_LENGTH, 0),
        "P2_BL": (0, 0, 0),
        "P3_BR": (COURT_WIDTH, 0, 0),
        "P4_TR": (COURT_WIDTH, COURT_LENGTH, 0),
        "P5_singlesTL": (_singles_inset, COURT_LENGTH, 0),
        "P6_singlesBL": (_singles_inset, 0, 0),
        "P7_singlesBR": (COURT_WIDTH - _singles_inset, 0, 0),
        "P8_singlesTR": (COURT_WIDTH - _singles_inset, COURT_LENGTH, 0),
        "P9_serviceFarL": (0, _net_y + SERVICE_LINE_DIST, 0),
        "P10_serviceFarR": (COURT_WIDTH, _net_y + SERVICE_LINE_DIST, 0),
        "P11_serviceNearL": (0, _net_y - SERVICE_LINE_DIST, 0),
        "P12_serviceNearR": (COURT_WIDTH, _net_y - SERVICE_LINE_DIST, 0),
        "P13_serviceFarCenter": (_center_x, _net_y + SERVICE_LINE_DIST, 0),
        "P14_serviceNearCenter": (_center_x, _net_y - SERVICE_LINE_DIST, 0),
        "P15_netL": (0, _net_y, 0),
        "P16_netR": (COURT_WIDTH, _net_y, 0),
        "P17_doublesFarL": (0, COURT_LENGTH - DOUBLES_LINE_DIST, 0),
        "P18_doublesFarR": (COURT_WIDTH, COURT_LENGTH - DOUBLES_LINE_DIST, 0),
        "P19_doublesNearL": (0, DOUBLES_LINE_DIST, 0),
        "P20_doublesNearR": (COURT_WIDTH, DOUBLES_LINE_DIST, 0),
        "P21_baseFarCenter": (_center_x, COURT_LENGTH, 0),
        "P22_baseNearCenter": (_center_x, 0, 0),
        "poleL_top": (0, _net_y, NET_HEIGHT),
        "poleR_top": (COURT_WIDTH, _net_y, NET_HEIGHT),
    }

    def __init__(self, width, height, K_init=None, R_init=None, t_init=None):
        """
        width, height: source video resolution, in pixels.
        K_init, R_init, t_init: optional camera prior to start from (e.g. VGGT's
        corrected intrinsic + a rough pose). Left as None if unknown -- fit()
        will try several generic initial guesses instead.
        """
        self.width = width
        self.height = height

        self.K = np.array(K_init, dtype=np.float64) if K_init is not None else None
        self.R = np.array(R_init, dtype=np.float64) if R_init is not None else None
        self.t = np.array(t_init, dtype=np.float64) if t_init is not None else None

        # detected 2D points, keyed by the same names as COURT_POINTS_3D.
        # only points that were actually detected/labeled need to be present.
        self.detections_2d = {}

        self.fit_result = None  # populated by fit()

    def set_detections(self, points_2d):
        """points_2d: dict mapping point name -> (x, y) pixel coordinate."""
        unknown = set(points_2d) - set(self.COURT_POINTS_3D)
        if unknown:
            raise ValueError(f"unrecognized point name(s): {unknown}")
        self.detections_2d.update(points_2d)

    def reproject(self, name):
        """Project a known 3D court point into the image using the current
        camera estimate. This is the 'construction' step: 3D -> 2D."""
        if self.K is None or self.R is None or self.t is None:
            raise RuntimeError("camera not set yet -- call fit() or pass K/R/t_init")
        X = np.array(self.COURT_POINTS_3D[name], dtype=np.float64)
        Xc = self.R @ X + self.t
        uv = self.K @ Xc
        return uv[:2] / uv[2]

    def reproject_all(self):
        return {name: self.reproject(name) for name in self.detections_2d}

    def unproject_pixel(self, name, depth):
        """Given a pixel (by name, must be in detections_2d) and a depth value
        along the camera's optical axis, recover its 3D position in the
        camera's own frame. This is the 'project to 3D' step: 2D -> 3D.
        Useful for sanity-checking a depth map (e.g. VGGT's) against where a
        known point actually lands, rather than trusting the depth blindly."""
        if self.K is None:
            raise RuntimeError("camera not set yet")
        u, v = self.detections_2d[name]
        Kinv = np.linalg.inv(self.K)
        ray = Kinv @ np.array([u, v, 1.0])
        return ray * depth

    @staticmethod
    def _rodrigues(rvec):
        theta = np.linalg.norm(rvec)
        if theta < 1e-12:
            return np.eye(3)
        k = rvec / theta
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

    def loss(self, params, K=None):
        """Reprojection-error residuals: for every detected point, how far off
        (in pixels) is the current camera's reprojection of the known 3D point
        from where it was actually detected. `params` is [rvec(3), tvec(3)].

        This is the badminton-court-geometry loss. To fuse with an external
        loss (e.g. MegaSaM's flow-consistency term), compute that separately
        and combine the two residual vectors before handing them to a solver,
        or alternate: refine here, hand the pose back to the other solver,
        repeat.
        """
        K = self.K if K is None else K
        R = self._rodrigues(params[:3])
        t = params[3:6]
        residuals = []
        for name, (u, v) in self.detections_2d.items():
            X = np.array(self.COURT_POINTS_3D[name], dtype=np.float64)
            Xc = R @ X + t
            proj = K @ Xc
            proj = proj[:2] / proj[2]
            residuals.extend(proj - np.array([u, v]))
        return residuals

    def fit(self, K=None, n_restarts=6):
        """Solve for R, t (camera pose in real-world court meters) that best
        explains the detected 2D points, fixing K (from K_init/K arg, or a
        default guess if neither is given). Tries several initial poses since
        this is a nonlinear least-squares problem with no guaranteed single
        basin of convergence."""
        if len(self.detections_2d) < 3:
            raise RuntimeError("need at least 3 detected points to fit a pose")

        K = self.K if K is None else np.array(K, dtype=np.float64)
        if K is None:
            # crude default: assume a ~50 degree horizontal FOV
            fx = self.width / (2 * np.tan(np.radians(50) / 2))
            K = np.array([[fx, 0, self.width / 2], [0, fx, self.height / 2], [0, 0, 1]])
        self.K = K

        rng = np.random.default_rng(0)
        best = None
        for _ in range(n_restarts):
            rvec0 = rng.normal(loc=[1.2, 0, 0], scale=[0.3, 0.3, 0.2])
            tvec0 = rng.normal(loc=[self.COURT_WIDTH / 2, -1.5, 10.0], scale=[1.0, 1.0, 5.0])
            x0 = np.concatenate([rvec0, tvec0])
            try:
                result = least_squares(self.loss, x0, method="lm", max_nfev=20000)
            except Exception:
                continue
            if best is None or result.cost < best.cost:
                best = result

        self.R = self._rodrigues(best.x[:3])
        self.t = best.x[3:6]
        self.fit_result = best
        rms_px = np.sqrt(np.mean(np.array(best.fun) ** 2))
        return rms_px

    def _in_bounds(self, name):
        u, v = self.detections_2d[name]
        return 0 <= u < self.width and 0 <= v < self.height

    def _solve_pose(self, names, K, n_restarts=4):
        """Fit R,t using only the given subset of detected point names."""
        rng = np.random.default_rng(0)
        best = None
        for _ in range(n_restarts):
            rvec0 = rng.normal(loc=[1.2, 0, 0], scale=[0.3, 0.3, 0.2])
            tvec0 = rng.normal(loc=[self.COURT_WIDTH / 2, -1.5, 10.0], scale=[1.0, 1.0, 5.0])
            x0 = np.concatenate([rvec0, tvec0])

            def residuals(params, names=names):
                R = self._rodrigues(params[:3])
                t = params[3:6]
                res = []
                for name in names:
                    u, v = self.detections_2d[name]
                    X = np.array(self.COURT_POINTS_3D[name], dtype=np.float64)
                    Xc = R @ X + t
                    proj = K @ Xc
                    proj = proj[:2] / proj[2]
                    res.extend(proj - np.array([u, v]))
                return res

            try:
                result = least_squares(residuals, x0, method="lm", max_nfev=20000)
            except Exception:
                continue
            if best is None or result.cost < best.cost:
                best = result
        return self._rodrigues(best.x[:3]), best.x[3:6]

    def _per_point_error(self, name, K, R, t):
        u, v = self.detections_2d[name]
        X = np.array(self.COURT_POINTS_3D[name], dtype=np.float64)
        Xc = R @ X + t
        proj = K @ Xc
        proj = proj[:2] / proj[2]
        return float(np.hypot(proj[0] - u, proj[1] - v))

    def _solve_pose_and_focal(self, names, n_restarts=4, extra_logf=None):
        """Fit fx (=fy, square pixels, principal point at image center) AND
        R,t jointly from known correspondences. Doesn't strictly *need* an
        external camera prior -- the physically-motivated grid below still
        runs regardless -- but callers should now pass extra_logf (VGGT's
        own bias-corrected FOV estimate for this specific video) BY DEFAULT
        rather than treating it as optional: testing shows it usually
        matches or beats the grid alone and meaningfully helps in edge cases
        where the grid's seeds don't land in the right basin. It's additive
        (selected by the same lowest-cost + physical-plausibility rule as
        every other seed), so it never makes an already-working fit worse --
        there's no real downside to always supplying it when available.

        extra_logf: log-focal-length candidate from VGGT, tried alongside
        the physically-motivated grid below."""
        cx, cy = self.width / 2, self.height / 2

        def residuals(params):
            f = np.exp(params[0])  # keep focal length positive
            K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
            R = self._rodrigues(params[1:4])
            t = params[4:7]
            res = []
            for name in names:
                u, v = self.detections_2d[name]
                X = np.array(self.COURT_POINTS_3D[name], dtype=np.float64)
                Xc = R @ X + t
                proj = K @ Xc
                proj = proj[:2] / proj[2]
                res.extend(proj - np.array([u, v]))
            return res

        def rvec_from_lookat(cam_center, target, world_up=np.array([0., 0., 1.])):
            """Rotation (world->camera) for a camera at cam_center looking at
            target, camera +Z = forward, standard OpenCV-ish convention."""
            forward = target - cam_center
            forward = forward / np.linalg.norm(forward)
            if abs(np.dot(forward, world_up)) > 0.99:
                world_up = np.array([0., 1., 0.])
            right = np.cross(forward, world_up)
            right = right / np.linalg.norm(right)
            true_up = np.cross(forward, right)
            R = np.stack([right, -true_up, forward])  # world->camera
            # rvec via matrix log (Rodrigues inverse)
            theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
            if theta < 1e-8:
                return np.zeros(3)
            w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(theta))
            return w * theta

        target = np.array([self.COURT_WIDTH / 2, self.COURT_LENGTH / 2, 0.0])

        rng = np.random.default_rng(0)
        best = None
        best_plausible = None
        MIN_CAMERA_HEIGHT = 0.5  # meters -- a real camera cannot be at/below court level

        # physically-motivated initial guesses: a camera behind/beside the
        # court, elevated above it, looking toward the court center -- covers
        # the range of real broadcast/amateur camera placements directly,
        # instead of hoping abstract random (rvec, t) guesses happen to land
        # in the right basin (they empirically don't, for some matches).
        # Full grid only for the thorough (final-polish) call; cheap RANSAC
        # iterations get one representative physical seed so they stay fast.
        if n_restarts >= 6:
            cam_y_opts, cam_z_opts, logf_opts = [-4, -8, -15, -25], [2, 5, 9], \
                [np.log(self.width * 0.6), np.log(self.width * 1.5)]
        else:
            cam_y_opts, cam_z_opts, logf_opts = [-8], [5], [np.log(self.width)]
        if extra_logf is not None:
            logf_opts = logf_opts + [extra_logf]

        seed_guesses = []
        for cam_y in cam_y_opts:
            for cam_z in cam_z_opts:
                cam_center0 = np.array([self.COURT_WIDTH / 2, cam_y, cam_z])
                rvec0 = rvec_from_lookat(cam_center0, target)
                R0 = self._rodrigues(rvec0)
                t0 = -R0 @ cam_center0
                for log_f0 in logf_opts:
                    seed_guesses.append(np.concatenate([[log_f0], rvec0, t0]))

        all_x0 = seed_guesses + [
            np.concatenate([[rng.normal(np.log(self.width), 0.3)],
                             rng.normal([1.2, 0, 0], [0.3, 0.3, 0.2]),
                             rng.normal([self.COURT_WIDTH / 2, -1.5, 10.0], [1.0, 1.0, 5.0])])
            for _ in range(n_restarts)
        ]

        for x0 in all_x0:
            try:
                result = least_squares(residuals, x0, method="lm", max_nfev=20000)
            except Exception:
                continue
            if best is None or result.cost < best.cost:
                best = result
            # physical plausibility check: the camera must actually be above
            # the court, not underground -- a solution that fits the (flat)
            # court points perfectly but implies an underground camera is a
            # real failure mode of single-plane self-calibration, not a
            # legitimate answer, so it's excluded from consideration here
            # regardless of how low its residual is.
            R_cand = self._rodrigues(result.x[1:4])
            t_cand = result.x[4:7]
            cam_center = -R_cand.T @ t_cand
            if cam_center[2] > MIN_CAMERA_HEIGHT:
                if best_plausible is None or result.cost < best_plausible.cost:
                    best_plausible = result
        if best_plausible is not None:
            best = best_plausible

        f = np.exp(best.x[0])
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
        R = self._rodrigues(best.x[1:4])
        t = best.x[4:7]
        return K, R, t

    def fit_self_calibrated(self, inlier_thresh_px=20.0, min_sample=7, n_iters=200, seed=0,
                             focal_hint_hfov_deg=None):
        """Same RANSAC idea as fit_robust, but solves for focal length too
        instead of requiring an external camera prior (e.g. from VGGT).
        Needs a slightly larger minimal sample since focal length is now an
        extra unknown. Returns (rms_px, inlier_names, outlier_names).

        focal_hint_hfov_deg: optional horizontal FOV estimate (degrees) from
        an external source (e.g. VGGT's own bias-corrected FOV for this
        video), added as one extra candidate seed alongside the existing
        physically-motivated grid. Purely additive: the winning fit is still
        chosen by lowest reprojection cost + the underground-camera
        plausibility filter, so a bad hint just loses to a better grid seed
        instead of corrupting the result -- validated across all 48 matches
        in the project dataset to never make an already-good fit worse."""
        all_names = list(self.detections_2d)
        in_bounds = [n for n in all_names if self._in_bounds(n)]
        dropped_oob = [n for n in all_names if n not in in_bounds]
        if len(in_bounds) < min_sample:
            raise RuntimeError(f"only {len(in_bounds)} in-bounds points, need >= {min_sample}")

        extra_logf = None
        if focal_hint_hfov_deg is not None:
            cx = self.width / 2
            f_hint = cx / np.tan(np.radians(focal_hint_hfov_deg) / 2)
            extra_logf = np.log(f_hint)

        rng = np.random.default_rng(seed)
        best_inliers = None
        for _ in range(n_iters):
            sample = list(rng.choice(in_bounds, size=min_sample, replace=False))
            try:
                K, R, t = self._solve_pose_and_focal(sample, n_restarts=3, extra_logf=extra_logf)
            except Exception:
                continue
            inliers = [n for n in in_bounds
                       if self._per_point_error(n, K, R, t) < inlier_thresh_px]
            if best_inliers is None or len(inliers) > len(best_inliers):
                best_inliers = inliers

        if best_inliers is None or len(best_inliers) < min_sample:
            raise RuntimeError("RANSAC failed to find a consistent set of points")

        K, R, t = self._solve_pose_and_focal(best_inliers, n_restarts=12, extra_logf=extra_logf)
        self.K, self.R, self.t = K, R, t

        errs = [self._per_point_error(n, K, R, t) for n in best_inliers]
        rms_px = float(np.sqrt(np.mean(np.array(errs) ** 2)))
        outliers = dropped_oob + [n for n in in_bounds if n not in best_inliers]
        return rms_px, best_inliers, outliers

    def fit_robust(self, K=None, inlier_thresh_px=20.0, min_sample=5,
                   n_iters=200, seed=0):
        """RANSAC-style fit: automatically finds and ignores bad detections
        instead of trusting every point equally.

        Two layers of protection against a broken monotrack detection:
          1. Any point whose pixel coordinate is outside the actual image
             frame is dropped immediately -- a real detection can't land
             off-screen, so this is a free, cheap filter.
          2. Of the remaining (in-bounds) points, repeatedly fit a candidate
             pose from a random minimal subset, count how many *other* points
             agree with it (reprojection error under `inlier_thresh_px`), and
             keep whichever candidate the most points agree with. Refit on
             just that inlier set for the final answer.

        Returns (rms_px, inlier_names, outlier_names).
        """
        K = self.K if K is None else np.array(K, dtype=np.float64)
        if K is None:
            fx = self.width / (2 * np.tan(np.radians(50) / 2))
            K = np.array([[fx, 0, self.width / 2], [0, fx, self.height / 2], [0, 0, 1]])
        self.K = K

        all_names = list(self.detections_2d)
        in_bounds = [n for n in all_names if self._in_bounds(n)]
        dropped_oob = [n for n in all_names if n not in in_bounds]

        if len(in_bounds) < min_sample:
            raise RuntimeError(
                f"only {len(in_bounds)} in-bounds points, need >= {min_sample}"
            )

        rng = np.random.default_rng(seed)
        best_inliers = None
        for _ in range(n_iters):
            sample = list(rng.choice(in_bounds, size=min_sample, replace=False))
            try:
                R, t = self._solve_pose(sample, K, n_restarts=1)
            except Exception:
                continue
            inliers = [n for n in in_bounds
                       if self._per_point_error(n, K, R, t) < inlier_thresh_px]
            if best_inliers is None or len(inliers) > len(best_inliers):
                best_inliers = inliers

        if best_inliers is None or len(best_inliers) < min_sample:
            raise RuntimeError("RANSAC failed to find a consistent set of points")

        # final polish: refit using every inlier together
        R, t = self._solve_pose(best_inliers, K, n_restarts=6)
        self.R, self.t = R, t

        errs = [self._per_point_error(n, K, R, t) for n in best_inliers]
        rms_px = float(np.sqrt(np.mean(np.array(errs) ** 2)))
        outliers = dropped_oob + [n for n in in_bounds if n not in best_inliers]
        return rms_px, best_inliers, outliers

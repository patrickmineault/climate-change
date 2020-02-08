import numpy as np
import pickle
from scipy import ndimage
from skimage.restoration import inpaint
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


def infill_large_regions(I, npixels=10000, precision=1000):
    """Infill large missing regions with Gaussian Process Regression on a ring  asdasdasdd
    around each missing region.

    Arguments:
        I: an image
        npixels: the minimum size of regions to infill.
        precision: the number of points to use in the GPR.

    Returns:
        A partially infilled image.
    """
    assert I.shape[0] == I.shape[1]
    xgrid, ygrid = np.meshgrid(np.arange(I.shape[1]),
                               np.arange(I.shape[0]))

    I_ = I.copy()

    # Exclude two pixels on the border during infilling.
    bad_regions, n_bad_regions = ndimage.label(
        ndimage.binary_dilation(np.isnan(I), iterations=2))

    # Use 5 pixel regions surrounding each hole.
    surround = ndimage.grey_dilation(bad_regions, size=5)
    counts, _ = np.histogram(bad_regions, np.arange(n_bad_regions + 1) - .5)

    for i in range(1, n_bad_regions):
        if counts[i] > npixels:
            # This is a big region, infill using the GPR method.
            surround_data = (surround == i) & (bad_regions == 0)
            xgrid_s, ygrid_s = xgrid[surround_data], ygrid[surround_data]

            # Take N_points points at random, fit a Gaussian process.
            subs = np.random.permutation(np.arange(len(xgrid_s)))[:precision]
            gp_kernel = Matern(length_scale=1,
                               length_scale_bounds=(.01, 100), nu=1.5)
            gpr = GaussianProcessRegressor(kernel=gp_kernel, normalize_y=True)

            X = np.concatenate((xgrid_s.reshape(-1, 1),
                                ygrid_s.reshape(-1, 1)), axis=1)
            gpr.fit(X[subs, :], I[surround_data][subs])
            xgrid_s, ygrid_s = xgrid[bad_regions == i], ygrid[bad_regions == i]
            X_ = np.concatenate((xgrid_s.reshape(-1, 1),
                                 ygrid_s.reshape(-1, 1)), axis=1)
            y_ = gpr.predict(X_)
            I_[bad_regions == i] = y_
    return I_


def infill_small_regions(I):
    """Infill small nan regions within an image using a tiling approach to save
    on memory.

    Arguments:
        I: the image

    Returns:
        The infilled image.
    """
    n_tiles = 4 # ntiles horizontally.
    assert I.shape[0] == I.shape[1]
    tile_size = I.shape[0] // (n_tiles - 1)
    tile_delta = tile_size // 2

    k = 0
    I_stack = np.ones(I.shape + (2, 2)) * np.nan
    for j in range(n_tiles * 2 - 1):
        for i in range(n_tiles * 2 - 1):
            dy = slice(tile_delta * j, tile_delta * (j + 2))
            dx = slice(tile_delta * i, tile_delta * (i + 2))
            S = I[dy, dx]
            M = ndimage.binary_dilation(np.isnan(S), iterations=2)
            image_inpainted = inpaint.inpaint_biharmonic(S, M, multichannel=False)
            I_stack[dy, dx, j % 2, i % 2] = image_inpainted
            k += 1
    return np.nanmean(np.nanmean(I_stack, axis=2), axis=2)


def estimate_ground_height(P):
    """Estimates ground height from a point cloud.

    Arguments:
      P: a point cloud, n x 4 matrix, the columns mean long, lat, altitude,
         point type. 2 is ground.

    Returns:
      A dict object containing an estimate of the ground height sampled at
      1024 x 1024.
    """
    assert P.shape[1] == 4
    precision = 1024

    xrg = np.linspace(P[:, 0].min(), P[:, 0].max(), precision + 1)
    yrg = np.linspace(P[:, 1].min(), P[:, 1].max(), precision + 1)

    val_idx = P[:, 3] == 2

    npoints, _, _ = np.histogram2d(P[val_idx, 0], P[val_idx, 1], [xrg, yrg])
    total_height, _, _ = np.histogram2d(P[val_idx, 0], P[val_idx, 1],
                                          [xrg, yrg], weights=P[val_idx, 2])
    mean_height = total_height / npoints

    mean_height_large = infill_large_regions(mean_height)
    mean_height_fine = infill_small_regions(mean_height_large)
    results = {'original': mean_height,
               'coarse': mean_height_large,
               'fine': mean_height_fine,
               'xgrid': .5 * (xrg[:-1] + xrg[1:]),
               'ygrid': .5 * (yrg[:-1] + yrg[1:])}
    return results

if __name__ == "__main__":
    P = np.load('mcgill.npy')
    results = estimate_ground_height(P)
    with open('mcgill_ground_estimates.pkl', 'wb') as f:
        pickle.dump(results, f)

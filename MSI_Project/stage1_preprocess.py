import os
import gc
import cv2
import time
import warnings
import numpy as np
import tifffile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from skimage.color import rgb2gray
from pewlib.io.imzml import ImzML

from config import MASTER_DIR, SKIP_FILENAME, MAX_WORKERS_STAGE1, CSV_PATH
from utils import should_skip_subdir, load_rgb_tiff, resize_larger_to_match_smaller, import_mass_list

def register_multichannel_images(fixed_image: np.ndarray, moving_image: np.ndarray, shrink_factor: int = 10):
    import SimpleITK as sitk

    def to_grayscale(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2: return image.astype(np.float32)
        elif image.ndim == 3 and image.shape[2] == 3: return rgb2gray(image).astype(np.float32)
        elif image.ndim == 3 and image.shape[2] == 1: return np.squeeze(image).astype(np.float32)
        raise ValueError(f"Unsupported image shape: {image.shape}")

    fixed_gray = to_grayscale(fixed_image)
    moving_gray = moving_image[:, :, 0].astype(float)

    fixed_sitk = sitk.Cast(sitk.GetImageFromArray(fixed_gray), sitk.sitkFloat32)
    moving_sitk = sitk.Cast(sitk.GetImageFromArray(moving_gray), sitk.sitkFloat32)
    fixed_sitk.SetSpacing([1.0, 1.0]); fixed_sitk.SetOrigin([0.0, 0.0])
    moving_sitk.SetSpacing([1.0, 1.0]); moving_sitk.SetOrigin([0.0, 0.0])

    fixed_shrunk = sitk.Shrink(fixed_sitk, [shrink_factor] * 2)
    moving_shrunk = sitk.Shrink(moving_sitk, [shrink_factor] * 2)

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    R.SetMetricSamplingPercentage(0.5)
    R.SetMetricSamplingStrategy(R.RANDOM)

    rigid_init = sitk.CenteredTransformInitializer(fixed_shrunk, moving_shrunk, sitk.Euler2DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)
    rigid_transform_obj = sitk.Euler2DTransform(rigid_init)

    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsRegularStepGradientDescent(learningRate=4.0, minStep=1e-4, numberOfIterations=200, gradientMagnitudeTolerance=1e-8)
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetShrinkFactorsPerLevel([2, 1])
    R.SetSmoothingSigmasPerLevel([1, 0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    R.SetInitialTransform(rigid_transform_obj, inPlace=True)
    rigid_transform_obj = R.Execute(fixed_shrunk, moving_shrunk)

    mesh_size = [3] * fixed_shrunk.GetDimension()
    bspline_init = sitk.BSplineTransformInitializer(fixed_shrunk, mesh_size)
    bspline_transform = sitk.BSplineTransform(fixed_shrunk.GetDimension(), 3)
    bspline_transform.SetParameters(bspline_init.GetParameters())

    composite_transform = sitk.CompositeTransform([rigid_transform_obj, bspline_transform])
    R.SetMetricSamplingStrategy(R.REGULAR)
    R.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=500, maximumNumberOfCorrections=5)
    R.SetShrinkFactorsPerLevel([1])
    R.SetSmoothingSigmasPerLevel([0])
    R.SetInitialTransform(composite_transform, inPlace=True)
    final_transform = R.Execute(fixed_shrunk, moving_shrunk)

    fixed_gray_np = sitk.GetArrayFromImage(fixed_sitk)
    tifffile.imwrite("FixedMSI.tif", (np.clip(fixed_gray_np / fixed_gray_np.max(), 0, 1) * 255).astype(np.uint8))

    moving_channels = [moving_image[:, :, i] for i in range(moving_image.shape[2])] if moving_image.ndim == 3 else [moving_image]
    registered_channels = []
    for ch in moving_channels:
        ch_sitk = sitk.GetImageFromArray(ch.astype(np.float32))
        ch_registered = sitk.Resample(ch_sitk, fixed_sitk, final_transform, sitk.sitkLinear, 0.0, ch_sitk.GetPixelID())
        registered_channels.append(sitk.GetArrayFromImage(ch_registered))

    registered_moving_full = np.stack(registered_channels, axis=-1) if len(registered_channels) > 1 else registered_channels[0]

    if registered_moving_full.ndim == 3:
        p1, p99 = np.percentile(registered_moving_full, (0, 99))
        rescaled = np.clip((registered_moving_full - p1) / (p99 - p1 + 1e-8), 0, 1)
        normalized = registered_moving_full / (np.max(registered_moving_full) + 1e-8)
        
        tifffile.imwrite("MovedMSI_multichannel.tif", np.moveaxis((rescaled * 255).astype(np.uint8), -1, 0), imagej=True, metadata={'axes': 'CYX'})
        tifffile.imwrite("MovedMSI.tif", (rescaled * 255).astype(np.uint8)[:, :, 0], photometric='minisblack')
        tifffile.imwrite("MovedMSIOLD.tif", (normalized * 255).astype(np.uint8)[:, :, 0], photometric='minisblack')

    return registered_moving_full, final_transform

def stage1_worker(subdir: Path):
    if should_skip_subdir(subdir): return True
    try:
        os.chdir(subdir)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # --- Load Images ---
            pre_files = [f for f in os.listdir('.') if f.lower().endswith('.tif') and 'pre' in f.lower() and not f.lower().startswith('premsi_')]
            post_files = [f for f in os.listdir('.') if f.lower().endswith('.tif') and 'post' in f.lower()]
            if not pre_files or not post_files: return True

            moving_image = load_rgb_tiff(pre_files[0])
            fixed_image = load_rgb_tiff(post_files[0])
            fixed_image, moving_image = resize_larger_to_match_smaller(fixed_image, moving_image, tolerance=0.2)

            for _ in range(10):
                try:
                    registered, transform = register_multichannel_images(fixed_image, moving_image, 10)
                    break
                except Exception:
                    pass

            imagepost = cv2.imread('FixedMSI.tif', cv2.IMREAD_GRAYSCALE)
            imagepre = cv2.imread('MovedMSIOLD.tif', cv2.IMREAD_GRAYSCALE)
            
            # Simplified Square Finding Logic (from your original code)
            square_found = False
            square_list = []
            combined = cv2.add(cv2.threshold(imagepost, 254, 255, cv2.THRESH_BINARY)[1], cv2.threshold(imagepre, 254, 255, cv2.THRESH_BINARY)[1])
            quick_morph = cv2.medianBlur(cv2.dilate(combined, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1), 29)
            contours, _ = cv2.findContours(quick_morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                approx = cv2.approxPolyDP(cnt, 0.015 * cv2.arcLength(cnt, True), True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    rect = cv2.minAreaRect(approx)
                    (cx, cy), (w, h), angle = rect
                    if 700 < w < 1500 and 700 < h < 1500 and abs(w - h) <= 200 and (cv2.contourArea(cnt) / (w * h)) > 0.85:
                        square_found, square_list = True, [rect]
                        break

            if not square_found: raise ValueError("Final Detection Failed.")

            (center, size, angle) = square_list[0]
            cx, cy, w, h = center[0], center[1], size[0], size[1]
            
            cv2.imwrite(SKIP_FILENAME, cv2.resize(cv2.drawContours(cv2.cvtColor(cv2.imread('FixedMSI.tif', cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR), [np.intp(cv2.boxPoints(square_list[0]))], 0, (0, 0, 255), 5), (0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA))

            imzml_files = [f for f in os.listdir(subdir) if f.lower().endswith('.imzml')]
            imzml = ImzML.from_file(os.path.join(subdir, imzml_files[0]))
            masses_of_interest = import_mass_list(CSV_PATH)
            
            data = imzml.extract_masses(masses_of_interest, mass_width_ppm=10).astype(np.float32)
            data_cellpose = imzml.extract_masses(760.5852, mass_width_ppm=10).astype(np.float32)
            tic = imzml.extract_tic().astype(np.float32)
            tic_safe = np.where(tic == 0, 1, tic).astype(np.float32)

            data /= tic_safe[:, :, np.newaxis] if data.ndim == 3 else tic_safe
            data_cellpose /= tic_safe[:, :, np.newaxis] if data_cellpose.ndim == 3 else tic_safe
            data[tic == 0] = 0; data_cellpose[tic == 0] = 0
            np.nan_to_num(data, copy=False, nan=0.0); np.nan_to_num(data_cellpose, copy=False, nan=0.0)

            resized_data = np.empty((int(h), int(w), data.shape[2]), dtype=np.float32)
            for i in range(data.shape[2]): resized_data[:, :, i] = cv2.resize(data[:, :, i], (int(w), int(h)), interpolation=cv2.INTER_LINEAR)
            del data; gc.collect()

            resized_data_cellpose = cv2.resize(data_cellpose, (int(w), int(h)), interpolation=cv2.INTER_LINEAR)
            np.nan_to_num(resized_data_cellpose, copy=False, nan=0.0)

            data_min, data_max = np.percentile(resized_data_cellpose, 0), np.percentile(resized_data_cellpose, 99)
            norm_8bit = (((resized_data_cellpose - data_min) / (data_max - data_min + 1e-8)) * 255).astype(np.uint8) if data_max >= data_min else np.zeros_like(resized_data_cellpose, dtype=np.uint8)
            
            resampled_array = np.moveaxis(tifffile.imread("MovedMSI_multichannel.tif"), 0, -1) if tifffile.imread("MovedMSI_multichannel.tif").ndim == 3 else tifffile.imread("MovedMSI_multichannel.tif")[:, :, np.newaxis]
            rotated_image = cv2.warpAffine(resampled_array, cv2.getRotationMatrix2D(center, angle, 1.0), (resampled_array.shape[1], resampled_array.shape[0]), flags=cv2.INTER_LINEAR)
            
            cropped_channels = [cv2.getRectSubPix(rotated_image[:, :, c], (int(w), int(h)), center) for c in range(rotated_image.shape[2])]
            cropped_array = np.stack(cropped_channels, axis=-1)

            stage1_npz = subdir / "PREMSI_stage1_ready.npz"
            temp_npz = str(stage1_npz).replace(".npz", "_temp.npz")
            np.savez_compressed(temp_npz, imgs_cp=cropped_array, resized_data=resized_data, image_for_cellpose=norm_8bit.copy(), masses_of_interest=np.array(masses_of_interest, dtype=float))
            os.replace(temp_npz, stage1_npz)
            tifffile.imwrite("PREMSI_stage1_ref_image.tif", norm_8bit.copy().astype(np.uint8))
            return True

    except Exception as e:
        print(f"[S1] ERROR in {subdir.name}: {e}")
        return False

def run_stage1_preprocess_parallel():
    t0 = time.time()
    subdirs = [d for d in MASTER_DIR.iterdir() if d.is_dir()]
    with ProcessPoolExecutor(max_workers=MAX_WORKERS_STAGE1) as executor:
        results = list(executor.map(stage1_worker, subdirs))
    
    failed_dirs = [subdirs[i] for i, res in enumerate(results) if res is False]
    if failed_dirs:
        with ProcessPoolExecutor(max_workers=1) as safe_executor:
            safe_executor.map(stage1_worker, failed_dirs)
    print(f"[S1] DONE in {time.time() - t0:.1f} s")

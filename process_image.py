import SimpleITK as sitk
import yaml
import matplotlib.pyplot as plt
import sys
import copy

if len(sys.argv) < 2:
  sys.stdout.write(f"Usage: {sys.argv[0]} <ConfigPath>")
  sys.exit(1)

# reading configuration file
configs = open(sys.argv[1], "r").read()
configs = yaml.load(configs, yaml.FullLoader)

config_input = configs['input']
if config_input['image_type'] == 'static':
  image = sitk.ReadImage(config_input['data_path'], sitk.sitkFloat32)
elif config_input['image_type'] == 'MRI':
  reader = sitk.ImageSeriesReader()
  dicom_names = reader.GetGDCMSeriesFileNames(config_input['data_path'])
  reader.SetFileNames(dicom_names)
  image = reader.Execute()

origin_data = sitk.GetArrayFromImage(image)

print(sitk.GetArrayFromImage(image).shape)

def imshow(mask, mask_title):
  # plot
  mask_data = sitk.GetArrayFromImage(mask)

  origin_data_draw = copy.deepcopy(origin_data)
  mask_data_draw = copy.deepcopy(mask_data)
  if len(mask_data.shape) > 2:
    origin_data_draw = origin_data_draw[configs['3D']['view_idx']]
    mask_data_draw = mask_data_draw[configs['3D']['view_idx']]

  fig, ax = plt.subplots(figsize=(12, 6), ncols=2)
  ax[0].imshow(origin_data_draw)
  ax[0].imshow(mask_data_draw, alpha=configs['output']['plot']['alpha'])
  ax[0].set_title("Origin image with mask")
  ax[1].imshow(mask_data_draw)
  ax[1].set_title("Mask of " + mask_title)
  plt.show()

# backup pixelID
pixelID = image.GetPixelID()
caster = sitk.CastImageFilter()
caster.SetOutputPixelType(sitk.sitkFloat32)
image = sitk.Cast(image, sitk.sitkFloat32)

# remove noise
config_smoothing = configs['smoothing']
smoothing = sitk.CurvatureAnisotropicDiffusionImageFilter()
smoothing.SetTimeStep(config_smoothing['timestep'])
smoothing.SetNumberOfIterations(config_smoothing['n_iters'])
smoothing.SetConductanceParameter(config_smoothing['conductance_param'])
smoothingOutput = smoothing.Execute(image)

# gradient magnitude
config_gradient = configs['gradient_magnitude']
gradientMagnitude = sitk.GradientMagnitudeRecursiveGaussianImageFilter()
gradientMagnitude.SetSigma(config_gradient['sigma'])
gradientOutput = gradientMagnitude.Execute(smoothingOutput)

# 
config_sigmoid = configs['sigmoid']
sigmoid = sitk.SigmoidImageFilter()
sigmoid.SetOutputMinimum(config_sigmoid['min'])
sigmoid.SetOutputMaximum(config_sigmoid['max'])
sigmoid.SetAlpha(config_sigmoid['alpha'])
sigmoid.SetBeta(config_sigmoid['beta'])
sigmoidOutput = sigmoid.Execute(gradientOutput)

# fast marching
config_fastmarching = configs['fastmarching']
fastMarching = sitk.FastMarchingImageFilter()
fastMarching.SetTrialPoints(config_fastmarching['list_seed'])
fastMarching.SetStoppingValue(config_fastmarching['stopping_time'])
fastMarchingOutput = fastMarching.Execute(sigmoidOutput)

# binarizer
config_binarizer = configs['binarizer']
binarizer = sitk.BinaryThresholdImageFilter()
binarizer.SetLowerThreshold(config_binarizer['lower_threshold'])
binarizer.SetUpperThreshold(config_binarizer['upper_threshold'])
binarizer.SetInsideValue(config_binarizer['inside_value'])
binarizer.SetOutsideValue(config_binarizer['outside_value'])
binarizerOutput = binarizer.Execute(fastMarchingOutput)
# cast to origin type of image
binarizerOutput = sitk.Cast(binarizerOutput, image.GetPixelID())

imshow(binarizerOutput, "Fast Marching")

# geodesic active contour
config_gac = configs['gac']
gac = sitk.GeodesicActiveContourLevelSetImageFilter()
gac.SetPropagationScaling(config_gac['propagation_scaling']) # P
gac.SetCurvatureScaling(config_gac['curvature_scaling']) # 
gac.SetAdvectionScaling(config_gac['advection_scaling'])
gac.SetMaximumRMSError(config_gac['max_rmse'])
gac.SetNumberOfIterations(config_gac['n_iters'])
result = gac.Execute(binarizerOutput, gradientOutput)

sys.stdout.write(f"RMS Change: {gac.GetRMSChange()} \n")
sys.stdout.write(f"Elapsed Iterations: {gac.GetElapsedIterations()} \n")

imshow(result, "Fast Marching + Geodesic Active Contour")
plt.imsave(
  configs['output']['mask_path'],
  sitk.GetArrayFromImage(result)[configs['3D']['view_idx']]
)

sitk.WriteImage(result, configs['output']['mha_path'])

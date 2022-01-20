import SimpleITK as sitk
import yaml
import matplotlib.pyplot as plt

# reading configuration file
configs = open("configs\configs.yaml", "r").read()
configs = yaml.load(configs, yaml.FullLoader)

config_input = configs['input']
if config_input['image_type'] == 'static':
  image = sitk.ReadImage(config_input['data_path'], sitk.sitkFloat32)
elif config_input['image_type'] == 'MRI':
  reader = sitk.ImageSeriesReader()
  dicom_names = reader.GetGDCMSeriesFileNames(config_input['data_path'])
  reader.SetFileNames(dicom_names)
  image = reader.Execute()

def imshow(origin, mask):
  # plot
  origin_data = sitk.GetArrayFromImage(origin)
  mask_data = sitk.GetArrayFromImage(mask)

  plt.imsave(configs['output']['mask_path'], mask_data)
  plt.imshow(origin_data)
  plt.imshow(mask_data, alpha=configs['output']['plot']['alpha'])
  plt.show()

# backup pixelID
pixelID = image.GetPixelID()
caster = sitk.CastImageFilter()
caster.SetOutputPixelType(pixelID)

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

imshow(image, binarizerOutput)

# geodesic active contour
config_gac = configs['gac']
gac = sitk.GeodesicActiveContourLevelSetImageFilter()
gac.SetPropagationScaling(config_gac['propagation_scaling'])
gac.SetCurvatureScaling(config_gac['curvature_scaling'])
gac.SetAdvectionScaling(config_gac['advection_scaling'])
gac.SetMaximumRMSError(config_gac['max_rmse'])
gac.SetNumberOfIterations(config_gac['n_iters'])
result = gac.Execute(binarizerOutput, gradientOutput)

print("RMS Change: ", gac.GetRMSChange())
print("Elapsed Iterations: ", gac.GetElapsedIterations())

imshow(image, result)

sitk.WriteImage(result, configs['output']['mha_path'])

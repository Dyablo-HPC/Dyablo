# trace generated using paraview version 5.13.2
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 13

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'XDMF Reader'
double_mach_mainxmf = XDMFReader(registrationName='double_mach_main.xmf', FileNames=['./double_mach_main.xmf'])

# get animation scene
animationScene1 = GetAnimationScene()

# update animation scene based on data timesteps
animationScene1.UpdateAnimationUsingDataTimeSteps()

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')

# show data in view
double_mach_mainxmfDisplay = Show(double_mach_mainxmf, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
double_mach_mainxmfDisplay.Representation = 'Surface'

# reset view to fit data
renderView1.ResetCamera(False, 0.9)

#changing interaction mode based on data extents
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [2.0, 0.5, 13.4]
renderView1.CameraFocalPoint = [2.0, 0.5, 0.0]

# get the material library
materialLibrary1 = GetMaterialLibrary()

# show color bar/color legend
double_mach_mainxmfDisplay.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# get color transfer function/color map for 'e_tot'
e_totLUT = GetColorTransferFunction('e_tot')

# get opacity transfer function/opacity map for 'e_tot'
e_totPWF = GetOpacityTransferFunction('e_tot')

# get 2D transfer function for 'e_tot'
e_totTF2D = GetTransferFunction2D('e_tot')

# get layout
layout1 = GetLayout()

# split cell
layout1.SplitVertical(0, 0.5)

# set active view
SetActiveView(None)

# Create a new 'Render View'
renderView2 = CreateView('RenderView')
renderView2.AxesGrid = 'Grid Axes 3D Actor'
renderView2.StereoType = 'Crystal Eyes'
renderView2.CameraFocalDisk = 1.0
renderView2.LegendGrid = 'Legend Grid Actor'
renderView2.PolarGrid = 'Polar Grid Actor'
renderView2.BackEnd = 'OSPRay raycaster'
renderView2.OSPRayMaterialLibrary = materialLibrary1

# assign view to a particular cell in the layout
AssignViewToLayout(view=renderView2, layout=layout1, hint=2)

# link cameras in two views
AddCameraLink(renderView2, renderView1, 'CameraLink0')

# set active view
SetActiveView(renderView1)

# set active view
SetActiveView(renderView2)

# set active source
SetActiveSource(double_mach_mainxmf)

# show data in view
double_mach_mainxmfDisplay_1 = Show(double_mach_mainxmf, renderView2, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
double_mach_mainxmfDisplay_1.Representation = 'Surface'

# show color bar/color legend
double_mach_mainxmfDisplay_1.SetScalarBarVisibility(renderView2, True)

# reset view to fit data
renderView2.ResetCamera(False, 0.9)

# set active view
SetActiveView(renderView1)

# Hide orientation axes
renderView1.OrientationAxesVisibility = 0

# set active view
SetActiveView(renderView2)

# Hide orientation axes
renderView2.OrientationAxesVisibility = 0

animationScene1.GoToLast()

# set active view
SetActiveView(renderView1)

# set scalar coloring
ColorBy(double_mach_mainxmfDisplay, ('CELLS', 'rho'))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(e_totLUT, renderView1)

# rescale color and/or opacity maps used to include current data range
double_mach_mainxmfDisplay.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
double_mach_mainxmfDisplay.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'rho'
rhoLUT = GetColorTransferFunction('rho')

# get opacity transfer function/opacity map for 'rho'
rhoPWF = GetOpacityTransferFunction('rho')

# get 2D transfer function for 'rho'
rhoTF2D = GetTransferFunction2D('rho')

# set active view
SetActiveView(renderView2)

# set scalar coloring
ColorBy(double_mach_mainxmfDisplay_1, ('CELLS', 'level'))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(e_totLUT, renderView2)

# rescale color and/or opacity maps used to include current data range
double_mach_mainxmfDisplay_1.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
double_mach_mainxmfDisplay_1.SetScalarBarVisibility(renderView2, True)

# get color transfer function/color map for 'level'
levelLUT = GetColorTransferFunction('level')

# get opacity transfer function/opacity map for 'level'
levelPWF = GetOpacityTransferFunction('level')

# get 2D transfer function for 'level'
levelTF2D = GetTransferFunction2D('level')

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
levelLUT.ApplyPreset('Plasma (matplotlib)', True)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
levelLUT.ApplyPreset('Magma (matplotlib)', True)

# get color legend/bar for levelLUT in view renderView2
levelLUTColorBar = GetScalarBar(levelLUT, renderView2)

# change scalar bar placement
levelLUTColorBar.WindowLocation = 'Any Location'
levelLUTColorBar.Position = [0.9090086011770032, 0.40394973070017953]

# change scalar bar placement
levelLUTColorBar.Position = [0.9379809868718878, 0.31597845601436264]

# set active view
SetActiveView(renderView1)

# get color legend/bar for rhoLUT in view renderView1
rhoLUTColorBar = GetScalarBar(rhoLUT, renderView1)

# change scalar bar placement
rhoLUTColorBar.WindowLocation = 'Any Location'
rhoLUTColorBar.Position = [0.9402444545043005, 0.3064516129032258]
rhoLUTColorBar.ScalarBarLength = 0.33

# change scalar bar placement
rhoLUTColorBar.Position = [0.9370755998189224, 0.3118279569892473]
rhoLUTColorBar.ScalarBarLength = 0.3300000000000001

# layout/tab size in pixels
layout1.SetSize(2209, 1116)

# current camera placement for renderView1
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [2.148645855192791, 0.4758909734063591, 7.96522841660369]
renderView1.CameraFocalPoint = [2.148645855192791, 0.4758909734063591, 0.0]
renderView1.CameraParallelScale = 0.5443976317709298

# current camera placement for renderView2
renderView2.InteractionMode = '2D'
renderView2.CameraPosition = [2.148645855192791, 0.4758909734063591, 7.96522841660369]
renderView2.CameraFocalPoint = [2.148645855192791, 0.4758909734063591, 0.0]
renderView2.CameraParallelScale = 0.5443976317709298

# save screenshot
SaveScreenshot(filename='./double_mach_result.png', viewOrLayout=layout1, location=16, SaveAllViews=1,
    ImageResolution=[2210, 1116],
    FontScaling='Do not scale fonts')

#================================================================
# addendum: following script captures some of the application
# state to faithfully reproduce the visualization during playback
#================================================================

#--------------------------------
# saving layout sizes for layouts

# layout/tab size in pixels
layout1.SetSize(2209, 1116)

#-----------------------------------
# saving camera placements for views

# current camera placement for renderView1
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [2.148645855192791, 0.4758909734063591, 7.96522841660369]
renderView1.CameraFocalPoint = [2.148645855192791, 0.4758909734063591, 0.0]
renderView1.CameraParallelScale = 0.5443976317709298

# current camera placement for renderView2
renderView2.InteractionMode = '2D'
renderView2.CameraPosition = [2.148645855192791, 0.4758909734063591, 7.96522841660369]
renderView2.CameraFocalPoint = [2.148645855192791, 0.4758909734063591, 0.0]
renderView2.CameraParallelScale = 0.5443976317709298


##--------------------------------------------
## You may need to add some code at the end of this python script depending on your usage, eg:
#
## Render all views to see them appears
# RenderAllViews()
#
## Interact with the view, usefull when running from pvpython
# Interact()
#
## Save a screenshot of the active view
# SaveScreenshot("path/to/screenshot.png")
#
## Save a screenshot of a layout (multiple splitted view)
# SaveScreenshot("path/to/screenshot.png", GetLayout())
#
## Save all "Extractors" from the pipeline browser
# SaveExtracts()
#
## Save a animation of the current active view
# SaveAnimation()
#
## Please refer to the documentation of paraview.simple
## https://www.paraview.org/paraview-docs/latest/python/paraview.simple.html
##--------------------------------------------

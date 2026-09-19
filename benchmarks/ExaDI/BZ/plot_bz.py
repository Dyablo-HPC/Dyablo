# trace generated using paraview version 5.13.2
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 13

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()
import os

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')

# destroy renderView1
Delete(renderView1)
del renderView1

# load state
LoadState('./visu_BZ.pvsm')

# find view
renderView1 = FindViewOrCreate('RenderView1', viewtype='RenderView')

# set active view
SetActiveView(renderView1)

# find view
renderView3 = FindViewOrCreate('RenderView3', viewtype='RenderView')

# set active view
SetActiveView(renderView3)

# find source
test_BZ_mainxmf = FindSource('test_BZ_main.xmf')

# set active source
SetActiveSource(test_BZ_mainxmf)

# get color transfer function/color map for 'level'
levelLUT = GetColorTransferFunction('level')

# get opacity transfer function/opacity map for 'level'
levelPWF = GetOpacityTransferFunction('level')

# get 2D transfer function for 'level'
levelTF2D = GetTransferFunction2D('level')

# get display properties
test_BZ_mainxmfDisplay = GetDisplayProperties(test_BZ_mainxmf, view=renderView3)

# rescale color and/or opacity maps used to exactly fit the current data range
test_BZ_mainxmfDisplay.RescaleTransferFunctionToDataRange(False, True)

# Rescale transfer function
levelLUT.RescaleTransferFunction(2.0, 10.0)

# Rescale transfer function
levelPWF.RescaleTransferFunction(2.0, 10.0)

# Rescale 2D transfer function
levelTF2D.RescaleTransferFunction(2.0, 10.0, 0.0, 1.0)

# find view
renderView2 = FindViewOrCreate('RenderView2', viewtype='RenderView')

# set active view
SetActiveView(renderView2)

# get color transfer function/color map for 'b'
bLUT = GetColorTransferFunction('b')

# get color legend/bar for bLUT in view renderView2
bLUTColorBar = GetScalarBar(bLUT, renderView2)

# get display properties
test_BZ_mainxmfDisplay_1 = GetDisplayProperties(test_BZ_mainxmf, view=renderView2)

# get opacity transfer function/opacity map for 'b'
bPWF = GetOpacityTransferFunction('b')

# get 2D transfer function for 'b'
bTF2D = GetTransferFunction2D('b')

# change scalar bar placement
bLUTColorBar.WindowLocation = 'Any Location'
bLUTColorBar.Position = [0.8293051359516617, 0.2795698924731183]
bLUTColorBar.ScalarBarLength = 0.3299999999999998

# Hide orientation axes
renderView2.OrientationAxesVisibility = 0

# get the material library
materialLibrary1 = GetMaterialLibrary()

# set active view
SetActiveView(renderView1)

# get display properties
test_BZ_mainxmfDisplay_2 = GetDisplayProperties(test_BZ_mainxmf, view=renderView1)

# get color transfer function/color map for 'a'
aLUT = GetColorTransferFunction('a')

# get opacity transfer function/opacity map for 'a'
aPWF = GetOpacityTransferFunction('a')

# get 2D transfer function for 'a'
aTF2D = GetTransferFunction2D('a')

# get color legend/bar for aLUT in view renderView1
aLUTColorBar = GetScalarBar(aLUT, renderView1)

# change scalar bar placement
aLUTColorBar.WindowLocation = 'Any Location'
aLUTColorBar.Position = [0.8205128205128205, 0.28315412186379924]
aLUTColorBar.ScalarBarLength = 0.3299999999999999

# Hide orientation axes
renderView1.OrientationAxesVisibility = 0

# find view
renderView4 = FindViewOrCreate('RenderView4', viewtype='RenderView')

# set active view
SetActiveView(renderView4)

# get display properties
test_BZ_mainxmfDisplay_3 = GetDisplayProperties(test_BZ_mainxmf, view=renderView4)

# get color transfer function/color map for 'c'
cLUT = GetColorTransferFunction('c')

# get opacity transfer function/opacity map for 'c'
cPWF = GetOpacityTransferFunction('c')

# get 2D transfer function for 'c'
cTF2D = GetTransferFunction2D('c')

# get color legend/bar for cLUT in view renderView4
cLUTColorBar = GetScalarBar(cLUT, renderView4)

# change scalar bar placement
cLUTColorBar.WindowLocation = 'Any Location'
cLUTColorBar.Position = [0.8174962292609352, 0.3393177737881508]
cLUTColorBar.ScalarBarLength = 0.3299999999999997

# Hide orientation axes
renderView4.OrientationAxesVisibility = 0

# set active view
SetActiveView(renderView3)

# Hide orientation axes
renderView3.OrientationAxesVisibility = 0

# get color legend/bar for levelLUT in view renderView3
levelLUTColorBar = GetScalarBar(levelLUT, renderView3)

# change scalar bar placement
levelLUTColorBar.WindowLocation = 'Any Location'
levelLUTColorBar.Position = [0.824773413897281, 0.3177737881508079]
levelLUTColorBar.ScalarBarLength = 0.3300000000000002

# link cameras in two views
AddCameraLink(renderView3, renderView4, 'CameraLink0')

# set active view
SetActiveView(renderView4)

# set active view
SetActiveView(renderView3)

# link cameras in two views
AddCameraLink(renderView3, renderView2, 'CameraLink1')

# set active view
SetActiveView(renderView2)

# set active view
SetActiveView(renderView3)

# link cameras in two views
AddCameraLink(renderView3, renderView1, 'CameraLink2')

# set active view
SetActiveView(renderView1)

# set active view
SetActiveView(renderView3)

renderView3.ResetActiveCameraToNegativeZ()

# reset view to fit data
renderView3.ResetCamera(False, 0.9)

# get layout
layout1 = GetLayout()

# layout/tab size in pixels
layout1.SetSize(1326, 1116)

# current camera placement for renderView1
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [0.619332203647281, 0.4949220338873498, 2.7320508075688776]
renderView1.CameraFocalPoint = [0.619332203647281, 0.4949220338873498, 0.0]
renderView1.CameraParallelScale = 0.584385769575659

# current camera placement for renderView2
renderView2.InteractionMode = '2D'
renderView2.CameraPosition = [0.619332203647281, 0.4949220338873498, 2.7320508075688776]
renderView2.CameraFocalPoint = [0.619332203647281, 0.4949220338873498, 0.0]
renderView2.CameraParallelScale = 0.584385769575659

# current camera placement for renderView3
renderView3.InteractionMode = '2D'
renderView3.CameraPosition = [0.619332203647281, 0.4949220338873498, 2.7320508075688776]
renderView3.CameraFocalPoint = [0.619332203647281, 0.4949220338873498, 0.0]
renderView3.CameraParallelScale = 0.584385769575659

# current camera placement for renderView4
renderView4.InteractionMode = '2D'
renderView4.CameraPosition = [0.619332203647281, 0.4949220338873498, 2.7320508075688776]
renderView4.CameraFocalPoint = [0.619332203647281, 0.4949220338873498, 0.0]
renderView4.CameraParallelScale = 0.584385769575659

# save animation
if not os.path.exists('render'):
  os.mkdir('render')
SaveAnimation(filename='render/img.png', viewOrLayout=layout1, location=16, SaveAllViews=1,
    ImageResolution=[1326, 1116],
    FontScaling='Do not scale fonts',
    FrameWindow=[0, 1000000])

#================================================================
# addendum: following script captures some of the application
# state to faithfully reproduce the visualization during playback
#================================================================

#--------------------------------
# saving layout sizes for layouts

# layout/tab size in pixels
layout1.SetSize(1326, 1116)

#-----------------------------------
# saving camera placements for views

# current camera placement for renderView1
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [0.619332203647281, 0.4949220338873498, 2.7320508075688776]
renderView1.CameraFocalPoint = [0.619332203647281, 0.4949220338873498, 0.0]
renderView1.CameraParallelScale = 0.584385769575659

# current camera placement for renderView2
renderView2.InteractionMode = '2D'
renderView2.CameraPosition = [0.619332203647281, 0.4949220338873498, 2.7320508075688776]
renderView2.CameraFocalPoint = [0.619332203647281, 0.4949220338873498, 0.0]
renderView2.CameraParallelScale = 0.584385769575659

# current camera placement for renderView3
renderView3.InteractionMode = '2D'
renderView3.CameraPosition = [0.619332203647281, 0.4949220338873498, 2.7320508075688776]
renderView3.CameraFocalPoint = [0.619332203647281, 0.4949220338873498, 0.0]
renderView3.CameraParallelScale = 0.584385769575659

# current camera placement for renderView4
renderView4.InteractionMode = '2D'
renderView4.CameraPosition = [0.619332203647281, 0.4949220338873498, 2.7320508075688776]
renderView4.CameraFocalPoint = [0.619332203647281, 0.4949220338873498, 0.0]
renderView4.CameraParallelScale = 0.584385769575659


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

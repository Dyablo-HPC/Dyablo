import sys
from paraview.simple import *

if( len(sys.argv) < 3 ):
    print("./render_pvsm.py <input>.pvsm <output>.png")
    exit -1

pvsm_filename = sys.argv[1]
image_filename = sys.argv[2]

servermanager.LoadState(pvsm_filename)
SetActiveView(GetRenderView())
#Render()
WriteImage(image_filename)

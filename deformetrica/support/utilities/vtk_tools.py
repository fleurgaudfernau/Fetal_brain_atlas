import sys
import vtk
import numpy as np
import os.path as op


def screenshot_vtk(file, name, overwrite = True):

    if not overwrite and op.exists(name):
        return

    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(file)
    reader.Update()
    #reader.GetOutput()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData( reader.GetOutput() )

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    #actor.GetProperty().SetColor(ageometry_color)

    # create a rendering window and renderer
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.SetSize(600,600)
    ren.SetBackground( 1, 1, 1)
 
    # create a renderwindowinteractor
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    ren.AddActor(actor)
    
    # render
    renWin.Render()
   
    if not name.endswith('.png'): name += '.png'

    grabber = vtk.vtkWindowToImageFilter()
    grabber.SetInput( renWin )
    grabber.SetMagnification(3)
    grabber.Update()
    
    writer = vtk.vtkPNGWriter()
    writer.SetInput(grabber.GetOutput() )
    writer.SetFileName(name)
    writer.Write()


import numpy
import vtk
from .raw import points as to_vtk_points
from .raw import vertices as to_vtk_vertices
from numpy2vtk.exceptions import Numpy2VtkFormatException

def vertices(points, z_index=0):
    """
    Returns the VTK-representation of a number of vertices that are defined by the points array.

    Args:
        points (numpy.ndarray<float> or vtk.vtkPoints): The points that the mesh consist of.
            If it's a numpy array it should be of dimensions (n,2) or (n,3)
        z_index (float): The value the z-value of 2d-points is filled with (only applicable for (n,2) input arrays)

    Returns:
        vertices_data (vtk.vtkPolyData): VTK polydata representation of the vertices
    """
    if not (isinstance(points, numpy.ndarray) or isinstance(points, vtk.vtkPoints)):
        raise Numpy2VtkFormatException(
            'vertices needs numpy array or vtk.vtkPoints as input'
        )

    if isinstance(points, numpy.ndarray):
        vtk_points = to_vtk_points(points, z_index=z_index) if isinstance(points, numpy.ndarray) else points
    else:
        vtk_points = points

    number_of_points = vtk_points.GetNumberOfPoints()
    vertices = to_vtk_vertices(numpy.array(range(number_of_points), dtype=numpy.int))

    vertices_data = vtk.vtkPolyData()
    vertices_data.SetPoints(vtk_points)
    vertices_data.SetVerts(vertices)

    return vertices_data

import numpy
import vtk
from .raw import points as to_vtk_points
from .raw import vertices as to_vtk_vertices
from .raw import edges as to_vtk_edges
from numpy2vtk.exceptions import Numpy2VtkFormatException

def line(points, z_index=0, closed=False):
    """
    Returns the VTK-representation of a line that is build from the points in the numpy array.

    Args:
        points (numpy.ndarray<float> or vtk.vtkPoints): The points that the line consist of.
            If it's a numpy array it should be of dimensions (n,2) or (n,3)
        z_index (float): The value the z-value of 2d-points is filled with (only applicable for (n,2) input arrays)
        closed (bool): Whether the last point of the line should be connected with the first one

    Returns:
        line_data (vtk.vtkPolyData): VTK polydata representation of the line
    """
    if not (isinstance(points, numpy.ndarray) or isinstance(points, vtk.vtkPoints)):
        raise Numpy2VtkFormatException(
            'line needs numpy array or vtk.vtkPoints as input'
        )

    if isinstance(points, numpy.ndarray):
        vtk_points = to_vtk_points(points, z_index=z_index) if isinstance(points, numpy.ndarray) else points
    else:
        vtk_points = points

    number_of_points = vtk_points.GetNumberOfPoints()
    edges = map(lambda i: [i, i+1], range(0, number_of_points-1))
    if closed:
        edges.append([number_of_points-1, 0])
    edges = numpy.array(edges, dtype=numpy.int)

    vtk_lines = to_vtk_edges(edges)
    vtk_vertices = to_vtk_vertices(numpy.array(range(number_of_points), dtype=numpy.int))

    line_data = vtk.vtkPolyData()
    line_data.SetPoints(vtk_points)
    line_data.SetVerts(vtk_vertices)
    line_data.SetLines(vtk_lines)

    return line_data

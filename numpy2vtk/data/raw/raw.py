import numpy
import vtk
from numpy2vtk.exceptions import Numpy2VtkFormatException

def points(coordinates, z_index=0):
    """
    Returns the raw VTK-representation of the points in the passed numpy array

    Args:
        coordinates (numpy.ndarray<float>): numpy.ndarray of shape (n,2) or (n,3) that contains the points
        z_index (float): The value the z-value of 2d-points is filled with (only applicable for (n,2) input arrays)

    Returns:
        vtk_points (vtk.vtkPoints): VTK representation of the points
    """
    if not isinstance(coordinates, numpy.ndarray):
        raise Numpy2VtkFormatException(
            'points needs numpy array as input'
        )
    if len(coordinates.shape) != 2:
        raise Numpy2VtkFormatException(
            'points needs a two dimensional array as input, was {}-dimensional'.format(len(coordinates.shape))
        )
    if coordinates.shape[1] != 2 and coordinates.shape[1] != 3:
        raise Numpy2VtkFormatException(
            'points needs an array of nx2 or nx3 shape, was nx{}'.format(coordinates.shape[1])
        )

    vtk_points = vtk.vtkPoints()
    for p in coordinates:
        z_value = p[2] if coordinates.shape[1] == 3 else z_index
        vtk_points.InsertNextPoint([p[0], p[1], z_value])

    return vtk_points

def vertices(indices):
    """
    Maps a numpy ndarray of shape (n,) to an vtkCellArray of vertex indices

    Args:
        indices (numpy.ndarray<int>): A numpy.ndarray of shape (n,) of indices that defines the n vertices

    Returns:
        vtk_vertices (vtk.vtkCellArray): VTK representation of the vertices
    """
    if not isinstance(indices, numpy.ndarray):
        raise Numpy2VtkFormatException(
            'vertices needs numpy array as input'
        )
    if len(indices.shape) != 1:
        raise Numpy2VtkFormatException(
            'vertices needs a one dimensional array as input, was {}-dimensional'.format(len(indices.shape))
        )
    if indices.dtype != numpy.int:
        raise Numpy2VtkFormatException(
            'vertices need to be numpy array of type numpy.int'
        )

    vtk_vertices = vtk.vtkCellArray()
    for v in indices:
        vtk_vertices.InsertNextCell(1)
        vtk_vertices.InsertCellPoint(v)
    return vtk_vertices

def edges(indices):
    """
    Maps a numpy ndarray to an vtkCellArray of vtkLines

    Args:
        indices (numpy.ndarray<int>): A numpy.ndarray of shape (n,2) of indices that define n edges

    Returns:
        vtk_lines (vtk.vtkCellArray): VTK representation of the edges
    """
    if not isinstance(indices, numpy.ndarray):
        raise Numpy2VtkFormatException(
            'lines needs numpy array as input'
        )
    if len(indices.shape) != 2 or indices.shape[1] != 2:
        raise Numpy2VtkFormatException(
            'lines needs a nx2 ndarray as input'
        )
    if indices.dtype != numpy.int:
        raise Numpy2VtkFormatException(
            'lines needs to be numpy array of type numpy.int'
        )
    vtk_lines = vtk.vtkCellArray()
    for e in indices:
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, e[0])
        line.GetPointIds().SetId(1, e[1])
        vtk_lines.InsertNextCell(line)
    return vtk_lines

def polygons(indices):
    """
    Maps a numpy ndarray to an vtkCellArray of vtkPolygons

    Args:
        indices (numpy.ndarray<int>): A numpy.ndarray of shape (n,m) of indices that define n polygons with m points each

    Returns:
        vtk_polygons (vtk.vtkCellArray): VTK representation of the polygons
    """
    if not isinstance(indices, numpy.ndarray):
        raise Numpy2VtkFormatException(
            'polygons needs numpy array as input'
        )
    if len(indices.shape) != 2:
        raise Numpy2VtkFormatException(
            'polygons needs a nxm ndarray as input'
        )
    if indices.dtype != numpy.int:
        raise Numpy2VtkFormatException(
            'polygons needs to be numpy array of type numpy.int'
        )

    number_of_polygons = indices.shape[0]
    poly_shape = indices.shape[1]
    vtk_polygons = vtk.vtkCellArray()
    for j in range(0, number_of_polygons):
        polygon = vtk.vtkPolygon()
        polygon.GetPointIds().SetNumberOfIds(poly_shape)
        for i in range(0, poly_shape):
            polygon.GetPointIds().SetId(i, indices[j, i])
        vtk_polygons.InsertNextCell(polygon)
    return vtk_polygons

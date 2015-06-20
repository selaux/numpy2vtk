import numpy
import vtk
from .raw import points as to_vtk_points
from .raw import vertices as to_vtk_vertices
from .raw import polygons as to_vtk_polygons
from numpy2vtk.exceptions import Numpy2VtkFormatException

def mesh(points, polys, z_index=0):
    """
    Returns the VTK-representation of a mesh that is build by creating the patches specified by points and polys.
    Points are the considered points and polys consists of an array of patches (which consist of indices into the
    points array).

    Args:
        points (numpy.ndarray<float> or vtk.vtkPoints): The points that the mesh consist of.
            If it's a numpy array it should be of dimensions (n,2) or (n,3)
        polys (numpy.ndarray<int>): Array of patches, should be of shape nxm for n patches with m points per patch
        z_index (float): The value the z-value of 2d-points is filled with (only applicable for (n,2) input arrays)

    Returns:
        poly_data (vtk.vtkPolyData): VTK polydata representation of the mesh
    """
    if not (isinstance(points, numpy.ndarray) or isinstance(points, vtk.vtkPoints)):
        raise Numpy2VtkFormatException(
            'mesh points needs to be numpy array or vtk.vtkPoints'
        )

    if isinstance(points, numpy.ndarray):
        vtk_points = to_vtk_points(points, z_index=z_index)
    else:
        vtk_points = points

    number_of_points = vtk_points.GetNumberOfPoints()
    if numpy.logical_or(polys > number_of_points-1, polys < 0).any():
        raise Numpy2VtkFormatException(
            'mesh polys references a point index that does not exist'
        )

    vtk_vertices = to_vtk_vertices(numpy.array(range(number_of_points), dtype=numpy.int))
    vtk_polygons = to_vtk_polygons(polys)

    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(vtk_points)
    poly_data.SetVerts(vtk_vertices)
    poly_data.SetPolys(vtk_polygons)

    return poly_data

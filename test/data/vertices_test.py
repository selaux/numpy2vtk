import numpy

from test import V2NUnitTest
from numpy2vtk.data.raw import points as to_vtk_points
from numpy2vtk.data import vertices
from numpy2vtk.exceptions import Numpy2VtkFormatException


class TestPVerticesData(V2NUnitTest):

    def test_vertices_with_2d_input_data(self):
        numpy_points = numpy.array([
            [1.0, 2.0],
            [3.0, 4.0]
        ])
        vtk_vertices = vertices(numpy_points)
        self.assertPoints(vtk_vertices.GetPoints(), [
            (1.0, 2.0, 0.0),
            (3.0, 4.0, 0.0)
        ])
        self.assertCellArray(vtk_vertices.GetVerts(), [
            (0,),
            (1,)
        ])

    def test_vertices_with_2d_input_data_and_z_index(self):
        numpy_points = numpy.array([
            [1.0, 2.0],
            [3.0, 4.0]
        ])
        vtk_vertices = vertices(numpy_points, z_index=1.0)
        self.assertPoints(vtk_vertices.GetPoints(), [
            (1.0, 2.0, 1.0),
            (3.0, 4.0, 1.0)
        ])
        self.assertCellArray(vtk_vertices.GetVerts(), [
            (0,),
            (1,)
        ])

    def test_vertices_with_3d_input_data(self):
        numpy_points = numpy.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ])
        vtk_vertices = vertices(numpy_points, z_index=1.0)
        self.assertPoints(vtk_vertices.GetPoints(), [
            (1.0, 2.0, 3.0),
            (4.0, 5.0, 6.0),
            (7.0, 8.0, 9.0)
        ])
        self.assertCellArray(vtk_vertices.GetVerts(), [
            (0,),
            (1,),
            (2,)
        ])

    def test_vertices_with_3d_input_data(self):
        vtk_points = to_vtk_points(numpy.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]))
        vtk_vertices = vertices(vtk_points, z_index=1.0)
        self.assertEqual(vtk_vertices.GetPoints(), vtk_points)
        self.assertCellArray(vtk_vertices.GetVerts(), [
            (0,),
            (1,),
            (2,)
        ])

    def test_vertices_with_wrong_input_type(self):
        numpy_points = 'numpy array'
        with self.assertRaisesRegexp(
                Numpy2VtkFormatException, 'vertices needs numpy array or vtk.vtkPoints as input'):
            vertices(numpy_points)

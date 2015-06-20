import numpy

from test import V2NUnitTest
from numpy2vtk.data.raw import points
from numpy2vtk.data import mesh
from numpy2vtk.exceptions import Numpy2VtkFormatException

class TestMeshData(V2NUnitTest):

    def test_mesh_with_2d_input_data(self):
        numpy_points = numpy.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],
        ])
        numpy_polys = numpy.array([
            [0, 1, 2],
            [0, 2, 3],
        ], dtype=numpy.int)

        vtk_mesh = mesh(numpy_points, numpy_polys)

        self.assertPoints(vtk_mesh.GetPoints(), [
            (0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
            (1.0, 0.0, 0.0),
        ])
        self.assertCellArray(vtk_mesh.GetVerts(), [
            (0,),
            (1,),
            (2,),
            (3,),
        ])
        self.assertCellArray(vtk_mesh.GetPolys(), [
            (0, 1, 2),
            (0, 2, 3)
        ])

    def test_line_with_2d_input_data_and_z_index(self):
        numpy_points = numpy.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],
        ])
        numpy_polys = numpy.array([
            [0, 1, 2],
            [0, 2, 3],
        ], dtype=numpy.int)

        vtk_mesh = mesh(numpy_points, numpy_polys, z_index=1.0)

        self.assertPoints(vtk_mesh.GetPoints(), [
            (0.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
            (1.0, 1.0, 1.0),
            (1.0, 0.0, 1.0),
        ])
        self.assertCellArray(vtk_mesh.GetVerts(), [
            (0,),
            (1,),
            (2,),
            (3,),
        ])
        self.assertCellArray(vtk_mesh.GetPolys(), [
            (0, 1, 2),
            (0, 2, 3)
        ])

    def test_mesh_with_3d_input_data(self):
        numpy_points = numpy.array([
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 2.0],
            [1.0, 1.0, 3.0],
            [1.0, 0.0, 4.0],
        ])
        numpy_polys = numpy.array([
            [0, 1, 2],
            [0, 2, 3],
        ], dtype=numpy.int)

        vtk_mesh = mesh(numpy_points, numpy_polys)

        self.assertPoints(vtk_mesh.GetPoints(), [
            (0.0, 0.0, 1.0),
            (0.0, 1.0, 2.0),
            (1.0, 1.0, 3.0),
            (1.0, 0.0, 4.0),
        ])
        self.assertCellArray(vtk_mesh.GetVerts(), [
            (0,),
            (1,),
            (2,),
            (3,),
        ])
        self.assertCellArray(vtk_mesh.GetPolys(), [
            (0, 1, 2),
            (0, 2, 3)
        ])

    def test_mesh_with_vtk_points_input_data(self):
        vtk_points = points(numpy.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],
        ]))

        numpy_polys = numpy.array([
            [0, 1, 2],
            [0, 2, 3],
        ], dtype=numpy.int)

        vtk_mesh = mesh(vtk_points, numpy_polys)

        self.assertEqual(vtk_mesh.GetPoints(), vtk_points)
        self.assertCellArray(vtk_mesh.GetVerts(), [
            (0,),
            (1,),
            (2,),
            (3,),
        ])
        self.assertCellArray(vtk_mesh.GetPolys(), [
            (0, 1, 2),
            (0, 2, 3)
        ])

    def test_mesh_with_quadrant_elements(self):
        numpy_points = numpy.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],
            [1.0, 2.0],
            [0.0, 2.0],
        ])
        numpy_polys = numpy.array([
            [0, 1, 2, 3],
            [3, 2, 4, 5],
        ], dtype=numpy.int)

        vtk_mesh = mesh(numpy_points, numpy_polys)

        self.assertPoints(vtk_mesh.GetPoints(), [
            (0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 2.0, 0.0),
            (0.0, 2.0, 0.0),
        ])
        self.assertCellArray(vtk_mesh.GetVerts(), [
            (0,),
            (1,),
            (2,),
            (3,),
            (4,),
            (5,),
        ])
        self.assertCellArray(vtk_mesh.GetPolys(), [
            (0, 1, 2, 3),
            (3, 2, 4, 5)
        ])

    def test_mesh_with_point_index_that_does_not_exist(self):
        numpy_points = numpy.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],
        ])
        numpy_polys = numpy.array([
            [0, 1, 2],
            [0, 2, 4],
        ], dtype=numpy.int)
        with self.assertRaisesRegexp(
                Numpy2VtkFormatException, 'mesh polys references a point index that does not exist'):
            mesh(numpy_points, numpy_polys)

    def test_mesh_with_negative_point_index(self):
        numpy_points = numpy.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],
        ])
        numpy_polys = numpy.array([
            [0, 1, 2],
            [0, 2, -1],
        ], dtype=numpy.int)
        with self.assertRaisesRegexp(
                Numpy2VtkFormatException, 'mesh polys references a point index that does not exist'):
            mesh(numpy_points, numpy_polys)

    def test_mesh_with_wrong_point_input_type(self):
        numpy_points = 'numpy array'
        numpy_polys = numpy.array([
            [0, 1, 2],
            [0, 2, 3],
        ], dtype=numpy.int)
        with self.assertRaisesRegexp(
                Numpy2VtkFormatException, 'mesh points needs to be numpy array or vtk.vtkPoints'):
            mesh(numpy_points, numpy_polys)

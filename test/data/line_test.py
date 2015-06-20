import numpy

from test import V2NUnitTest
from numpy2vtk.data.raw import points
from numpy2vtk.data import line
from numpy2vtk.exceptions import Numpy2VtkFormatException

class TestLineData(V2NUnitTest):

    def test_line_with_2d_input_data(self):
        numpy_points = numpy.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ])
        vtk_line = line(numpy_points)

        self.assertPoints(vtk_line.GetPoints(), [
            (1.0, 2.0, 0.0),
            (3.0, 4.0, 0.0),
            (5.0, 6.0, 0.0)
        ])
        self.assertCellArray(vtk_line.GetVerts(), [
            (0,),
            (1,),
            (2,),
        ])
        self.assertCellArray(vtk_line.GetLines(), [
            (0, 1),
            (1, 2)
        ])

    def test_line_with_2d_input_data_and_z_index(self):
        numpy_points = numpy.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ])
        vtk_line = line(numpy_points, z_index=1.0)

        self.assertPoints(vtk_line.GetPoints(), [
            (1.0, 2.0, 1.0),
            (3.0, 4.0, 1.0),
            (5.0, 6.0, 1.0)
        ])
        self.assertCellArray(vtk_line.GetVerts(), [
            (0,),
            (1,),
            (2,),
        ])
        self.assertCellArray(vtk_line.GetLines(), [
            (0, 1),
            (1, 2)
        ])

    def test_line_with_3d_input_data(self):
        numpy_points = numpy.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ])
        vtk_line = line(numpy_points)

        self.assertPoints(vtk_line.GetPoints(), [
            (1.0, 2.0, 3.0),
            (4.0, 5.0, 6.0),
            (7.0, 8.0, 9.0)
        ])
        self.assertCellArray(vtk_line.GetVerts(), [
            (0,),
            (1,),
            (2,),
        ])
        self.assertCellArray(vtk_line.GetLines(), [
            (0, 1),
            (1, 2)
        ])

    def test_line_with_closed_param(self):
        numpy_points = numpy.array([
            [1.0, 2.0],
            [4.0, 5.0],
            [7.0, 8.0]
        ])
        vtk_line = line(numpy_points, closed=True)

        self.assertPoints(vtk_line.GetPoints(), [
            (1.0, 2.0, 0.0),
            (4.0, 5.0, 0.0),
            (7.0, 8.0, 0.0)
        ])
        self.assertCellArray(vtk_line.GetVerts(), [
            (0,),
            (1,),
            (2,),
        ])
        self.assertCellArray(vtk_line.GetLines(), [
            (0, 1),
            (1, 2),
            (2, 0)
        ])

    def test_line_with_vtk_points_input_data(self):
        vtk_points = points(numpy.array([
            [1.0, 2.0],
            [4.0, 5.0],
            [7.0, 8.0]
        ]))
        vtk_line = line(vtk_points)

        self.assertEqual(vtk_line.GetPoints(), vtk_points)
        self.assertCellArray(vtk_line.GetVerts(), [
            (0,),
            (1,),
            (2,),
        ])
        self.assertCellArray(vtk_line.GetLines(), [
            (0, 1),
            (1, 2)
        ])

    def test_line_with_wrong_input_type(self):
        numpy_points = 'numpy array'
        with self.assertRaisesRegexp(
                Numpy2VtkFormatException, 'line needs numpy array or vtk.vtkPoints as input'):
            line(numpy_points)

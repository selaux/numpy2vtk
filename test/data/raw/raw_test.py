import numpy

from test import V2NUnitTest
import numpy2vtk.data.raw as raw
from numpy2vtk.exceptions import Numpy2VtkFormatException

class RawPointsDataTest(V2NUnitTest):
    def test_2d_points(self):
        numpy_points = numpy.array([
            [1.0, 2.0],
            [3.0, 4.0]
        ])
        vtk_points = raw.points(numpy_points)
        self.assertPoints(vtk_points, [
            (1.0, 2.0, 0.0),
            (3.0, 4.0, 0.0),
        ])

    def test_2d_points_with_z_index(self):
        numpy_points = numpy.array([
            [1.0, 2.0],
            [3.0, 4.0]
        ])
        vtk_points = raw.points(numpy_points, z_index=1.0)
        self.assertPoints(vtk_points, [
            (1.0, 2.0, 1.0),
            (3.0, 4.0, 1.0),
        ])

    def test_3d_points(self):
        numpy_points = numpy.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ])
        vtk_points = raw.points(numpy_points, z_index=1.0)
        self.assertPoints(vtk_points, [
            (1.0, 2.0, 3.0),
            (4.0, 5.0, 6.0),
        ])

    def test_points_with_invalid_input_type(self):
        numpy_points = 'something'
        with self.assertRaisesRegexp(
                Numpy2VtkFormatException, 'points needs numpy array as input'):
            raw.points(numpy_points)

    def test_points_with_too_many_dimensions(self):
        numpy_points = numpy.array([
            [
                [1.0, 2.0]
            ],
            [
                [3.0, 4.0]
            ]
        ])

        with self.assertRaisesRegexp(
                Numpy2VtkFormatException, 'points needs a two dimensional array as input, was 3-dimensional'):
            raw.points(numpy_points)

    def test_points_with_invalid_dimensions(self):
        numpy_points = numpy.array([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0]
        ])

        with self.assertRaisesRegexp(
                Numpy2VtkFormatException, 'points needs an array of nx2 or nx3 shape, was nx4'):
            raw.points(numpy_points)


class RawVerticesDataTest(V2NUnitTest):
    def test_vertices(self):
        numpy_vertices = numpy.array([3, 4], dtype=numpy.int)
        vtk_vertices = raw.vertices(numpy_vertices)
        self.assertCellArray(vtk_vertices, [
            (3,),
            (4,),
        ])

    def test_vertices_with_invalid_input_type(self):
        numpy_vertices = 'something'
        with self.assertRaisesRegexp(
                Numpy2VtkFormatException, 'vertices needs numpy array as input'):
            raw.vertices(numpy_vertices)

    def test_vertices_with_too_many_dimensions(self):
        numpy_vertices = numpy.array([
            [1, 2],
            [3, 4]
        ], dtype=numpy.int)
        with self.assertRaisesRegexp(
                Numpy2VtkFormatException, 'vertices needs a one dimensional array as input, was 2-dimensional'):
            raw.vertices(numpy_vertices)

    def test_vertices_with_wrong_array_type(self):
        numpy_vertices = numpy.array([3, 4], dtype=numpy.float)
        with self.assertRaisesRegexp(
                Numpy2VtkFormatException, 'vertices need to be numpy array of type numpy.int'):
            raw.vertices(numpy_vertices)


class RawEdgesDataTest(V2NUnitTest):
    def test_edges(self):
        numpy_edges = numpy.array([
            [1, 2],
            [3, 4]
        ], dtype=numpy.int)
        vtk_edges = raw.edges(numpy_edges)
        self.assertCellArray(vtk_edges, [
            (1, 2),
            (3, 4),
        ])

    def test_edges_with_invalid_input_type(self):
        numpy_edges = 'something'
        with self.assertRaisesRegexp(
                Numpy2VtkFormatException, 'lines needs numpy array as input'):
            raw.edges(numpy_edges)

    def test_edges_with_too_many_dimensions(self):
        numpy_edges = numpy.array([
            [
                [1, 2]
            ],
            [
                [3, 4]
            ]
        ], dtype=numpy.int)
        with self.assertRaisesRegexp(
                Numpy2VtkFormatException, 'lines needs a nx2 ndarray as input'):
            raw.edges(numpy_edges)

    def test_edges_with_too_deep_dimensions(self):
        numpy_edges = numpy.array([
            [1, 2, 3],
            [3, 4, 4]
        ], dtype=numpy.int)
        with self.assertRaisesRegexp(
                Numpy2VtkFormatException, 'lines needs a nx2 ndarray as input'):
            raw.edges(numpy_edges)

    def test_edges_with_wrong_array_type(self):
        numpy_edges = numpy.array([
            [1, 2],
            [3, 4]
        ], dtype=numpy.float)
        with self.assertRaisesRegexp(
                Numpy2VtkFormatException, 'lines needs to be numpy array of type numpy.int'):
            raw.edges(numpy_edges)


class RawPolygonsDataTest(V2NUnitTest):
    def test_polygons(self):
        numpy_polygons = numpy.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ], dtype=numpy.int)
        vtk_polygons = raw.polygons(numpy_polygons)
        self.assertCellArray(vtk_polygons, [
            (1, 2, 3, 4),
            (5, 6, 7, 8),
        ])

    def test_polygons_with_invalid_input_type(self):
        numpy_polygons = 'something'
        with self.assertRaisesRegexp(
                Numpy2VtkFormatException, 'polygons needs numpy array as input'):
            raw.polygons(numpy_polygons)

    def test_polygons_with_too_many_dimensions(self):
        numpy_polygons = numpy.array([
            [
                [1, 2]
            ],
            [
                [3, 4]
            ]
        ], dtype=numpy.int)
        with self.assertRaisesRegexp(
                Numpy2VtkFormatException, 'polygons needs a nxm ndarray as input'):
            raw.polygons(numpy_polygons)

    def test_polygons_with_wrong_array_type(self):
        numpy_polygons = numpy.array([
            [1, 2],
            [3, 4]
        ], dtype=numpy.float)
        with self.assertRaisesRegexp(
                Numpy2VtkFormatException, 'polygons needs to be numpy array of type numpy.int'):
            raw.polygons(numpy_polygons)



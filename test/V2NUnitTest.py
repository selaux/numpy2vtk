import unittest
import vtk

class V2NUnitTest(unittest.TestCase):

    def assertCellArray(self, cell_array, expected):
        id_list = vtk.vtkIdList()
        got = []
        cell_array.InitTraversal()
        while cell_array.GetNextCell(id_list) != 0:
            got.append(
                tuple(
                    map(lambda n: id_list.GetId(n), range(0, id_list.GetNumberOfIds()))
                )
            )

        return self.assertEqual(got, expected)

    def assertPoints(self, points, expected):
        got = list(map(lambda n: points.GetPoint(n), range(0, points.GetNumberOfPoints())))
        return self.assertEqual(got, expected)


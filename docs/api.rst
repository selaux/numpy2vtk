=================
API Documentation
=================

Vtk Actors
==========

.. automodule:: numpy2vtk.actors
   :members:

Vtk PolyData
============

Vtk PolyData usually wraps multiple data types (e.g. Vertices, Edges) into a single data type. In VTK, they are used to
define how an actor is rendered. We return the data-only representation for an object here and define the visual options
in the actors module.

.. automodule:: numpy2vtk.data
   :members:

Raw Vtk Data
============

The data.raw module is the lowest representation of data in VTK. These are primarily used to update the data of
the actors.

.. automodule:: numpy2vtk.data.raw
   :members:

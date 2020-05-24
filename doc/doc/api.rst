API
===

shared
------

HydroParams
~~~~~~~~~~~

.. doxygenfile:: shared/HydroParams.h


SolverBase
~~~~~~~~~~

.. doxygenfile:: shared/SolverBase.h

muscl
-----

Compute functors
~~~~~~~~~~~~~~~~

.. doxygenfile:: muscl/ComputeDtHydroFunctor.h
.. doxygenfile:: muscl/ConvertToPrimitivesHydroFunctor.h
.. doxygenfile:: muscl/MarkCellsHydroFunctor.h
.. doxygenfile:: muscl/ReconstructGradientsHydroFunctor.h
.. doxygenfile:: muscl/UpdateRSSTHydroFunctor.h

SolverHydroMuscl
~~~~~~~~~~~~~~~~

.. doxygenfile:: muscl/SolverHydroMuscl.h

muscl-block
-----------

Compute functors
~~~~~~~~~~~~~~~~

.. doxygenfile:: muscl_block/ComputeDtHydroFunctor.h
.. doxygenfile:: muscl_block/ConvertToPrimitivesHydroFunctor.h
.. doxygenfile:: muscl_block/CopyCornerBlockCellData.h
.. doxygenfile:: muscl_block/CopyFaceBlockCellData.h
.. doxygenfile:: muscl_block/CopyInnerBlockCellData.h
.. doxygenfile:: muscl_block/MarkOctantsHydroFunctor.h
.. doxygenfile:: muscl_block/MusclBlockGodunovUpdateFunctor.h
.. doxygenfile:: muscl_block/MusclBlockSharedGodunovUpdateFunctor.h

SolverHydroMusclBlock
~~~~~~~~~~~~~~~~~~~~~

.. doxygenfile:: muscl_block/SolverHydroMusclBlock.h

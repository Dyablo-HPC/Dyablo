Build Doc
=========

This documentation should be built by the CI system and be made available as a gitlab-page (not possible with gitlab.maisondelasimulation.fr). Maybe we can just mirroring the doc on github.

This sphinx documentation can be build using CMake.

To build doxygen API documentation:

.. code-block:: bash

   mkdir build
   cd build
   cmake -DBUILD_DOC:BOOL=ON -DDOC:STRING=doxygen ..
   cd dyablo; make doc
   # the output is in doc/doxygen/html/index.html


To build sphinx/html documentation:		

.. code-block:: bash

   mkdir build
   cd build
   cmake -DBUILD_DOC:BOOL=ON -DDOC:STRING=html ..
   cd dyablo; make doc
   # the output web page is in build/doc/html/index.html


#pragma once

//! type of boundary condition (note that BC_COPY is only used in the
//! MPI version for inside boundary)
enum BoundaryConditionType {
  BC_UNDEFINED, 
  BC_REFLECTING,  /*!< reflecting border condition */
  BC_ABSORBING,   /*!< absorbing border condition */
  BC_PERIODIC,    /*!< periodic border condition */
  BC_COPY         /*!< only used in MPI parallelized version */
};

//! enum component index
enum ComponentIndex3D {
  IX = 0,
  IY = 1,
  IZ = 2
};

enum GravityType {
  GRAVITY_NONE       = 0,

  // General flags
  GRAVITY_CONSTANT = 1,
  GRAVITY_FIELD    = 2,

  // These are the flags to actually use in the code
  GRAVITY_CST_SCALAR = GRAVITY_CONSTANT,
  GRAVITY_CST_FIELD  = GRAVITY_CONSTANT | GRAVITY_FIELD
};

#pragma once

#define DYABLO_USE_AMRBlockForeachCell_group

#if defined(DYABLO_USE_AMRBlockForeachCell_group)

#include "AMRBlockForeachCell_group.h"
namespace dyablo{
namespace muscl_block{

using ForeachCell = AMRBlockForeachCell_group;

} // namespace muscl_block
} // namespace dyablo

#elif defined(DYABLO_USE_AMRBlockForeachCell_scratch)

#include "AMRBlockForeachCell_scratch.h"
namespace dyablo{
namespace muscl_block{

using ForeachCell = AMRBlockForeachCell_scratch;

} // namespace muscl_block
} // namespace dyablo

#else
#error "No ForeachCell defined"
#endif


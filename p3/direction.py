from enum import Enum
from point import Point

class Direction(Enum):
  NORTH = "⬆"
  EAST = "⮕"
  SOUTH = "⬇"
  WEST = "⬅"

  @classmethod
  def get_direction_val(self, direction):
    if direction == Direction.NORTH:
      return Point(0, 1)
    if direction == Direction.EAST:
      return Point(1, 0)
    if direction == Direction.SOUTH:
      return Point(0, -1)
    if direction == Direction.WEST:
      return Point(-1, 0)

    
  @classmethod
  def values(self):
    return [v for v in self]
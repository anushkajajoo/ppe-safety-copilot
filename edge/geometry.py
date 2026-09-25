"""
Small geometry helpers (pure Python, no dependencies).

point_in_polygon  : is a point inside a zone polygon?     (ray casting)
intersection_area : overlap area of two boxes
ioa               : "intersection over area" = how much of box A lies inside box B
"""
from typing import Sequence, Tuple

Box = Tuple[float, float, float, float]


def point_in_polygon(x: float, y: float, polygon: Sequence[Sequence[float]]) -> bool:
    """
    Ray casting: shoot a horizontal ray to the right from (x, y) and count how many
    polygon edges it crosses. Odd number of crossings = inside.
    """
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > y) != (yj > y):
            x_cross = xi + (y - yi) * (xj - xi) / (yj - yi)
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def intersection_area(a: Box, b: Box) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def ioa(a: Box, b: Box) -> float:
    """Fraction of box a that lies inside box b (0..1)."""
    aa = area(a)
    return intersection_area(a, b) / aa if aa > 0 else 0.0


def iou(a: Box, b: Box) -> float:
    """Intersection over union of two boxes (0..1) - 'are these the same object?'."""
    inter = intersection_area(a, b)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def center(b: Box) -> Tuple[float, float]:
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)

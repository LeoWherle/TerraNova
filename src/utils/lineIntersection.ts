export interface Point {
  x: number;
  y: number;
}

/**
 * Test whether two line segments (p1-p2) and (p3-p4) intersect.
 * Uses cross-product orientation test.
 */
export function segmentsIntersect(
  p1: Point,
  p2: Point,
  p3: Point,
  p4: Point,
): boolean {
  const d1x = p2.x - p1.x;
  const d1y = p2.y - p1.y;
  const d2x = p4.x - p3.x;
  const d2y = p4.y - p3.y;

  const denom = d1x * d2y - d1y * d2x;
  if (Math.abs(denom) < 1e-10) return false; // parallel

  const t = ((p3.x - p1.x) * d2y - (p3.y - p1.y) * d2x) / denom;
  const u = ((p3.x - p1.x) * d1y - (p3.y - p1.y) * d1x) / denom;

  return t >= 0 && t <= 1 && u >= 0 && u <= 1;
}

/**
 * Test whether any segment pair between two polylines intersects.
 */
export function polylinesIntersect(lineA: Point[], lineB: Point[]): boolean {
  for (let i = 0; i < lineA.length - 1; i++) {
    for (let j = 0; j < lineB.length - 1; j++) {
      if (segmentsIntersect(lineA[i], lineA[i + 1], lineB[j], lineB[j + 1])) {
        return true;
      }
    }
  }
  return false;
}

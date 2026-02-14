import connectionsData from "@/data/connections.json";

const connectionMatrix = connectionsData.connectionMatrix as Record<string, Record<string, number>>;

/** Pre-computed map: sourceCategory → set of target categories that accept it. */
const acceptableTargets = new Map<string, Set<string>>();

for (const [sourceCategory, targets] of Object.entries(connectionMatrix)) {
  const valid = new Set<string>();
  // Same-category is always valid
  valid.add(sourceCategory);
  for (const [targetCategory, count] of Object.entries(targets)) {
    if (count > 0) valid.add(targetCategory);
  }
  acceptableTargets.set(sourceCategory, valid);
}

/**
 * Check whether a node with `targetCategory` can accept a connection
 * from a handle with `sourceCategory`.
 */
export function isAcceptableTarget(sourceCategory: string, targetCategory: string): boolean {
  if (sourceCategory === targetCategory) return true;
  return acceptableTargets.get(sourceCategory)?.has(targetCategory) ?? false;
}

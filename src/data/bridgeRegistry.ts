/**
 * Bridge node registry — identifies nodes that accept inputs from
 * different categories than their own output category.
 * Derived from connections.json categoryBridges.
 */

import connectionsData from "./connections.json";
import { AssetCategory } from "@/schema/types";

interface BridgeInfo {
  from: AssetCategory;
  to: AssetCategory;
}

const CATEGORY_MAP: Record<string, AssetCategory> = {
  Density: AssetCategory.Density,
  Curve: AssetCategory.Curve,
  MaterialProvider: AssetCategory.MaterialProvider,
  Pattern: AssetCategory.Pattern,
  PositionProvider: AssetCategory.PositionProvider,
  Prop: AssetCategory.Prop,
  Scanner: AssetCategory.Scanner,
  Assignment: AssetCategory.Assignment,
  VectorProvider: AssetCategory.VectorProvider,
  EnvironmentProvider: AssetCategory.EnvironmentProvider,
  TintProvider: AssetCategory.TintProvider,
  BlockMask: AssetCategory.BlockMask,
  Directionality: AssetCategory.Directionality,
  Biome: AssetCategory.Biome,
};

// Build bridge type map from connections.json
const bridgeMap = new Map<string, BridgeInfo>();
for (const bridge of connectionsData.categoryBridges) {
  const from = CATEGORY_MAP[bridge.from];
  const to = CATEGORY_MAP[bridge.to];
  if (!from || !to) continue;
  for (const nodeType of bridge.bridgeTypes) {
    bridgeMap.set(nodeType, { from, to });
  }
}

/**
 * Check if a node type is a bridge node (connects two different categories).
 */
export function isBridgeNode(nodeType: string): boolean {
  return bridgeMap.has(nodeType);
}

/**
 * Get bridge info for a node type: which categories it connects.
 * Returns null if the node is not a bridge type.
 */
export function getBridgeInfo(nodeType: string): BridgeInfo | null {
  return bridgeMap.get(nodeType) ?? null;
}

import { useCallback } from "react";
import type { Edge, Connection } from "@xyflow/react";
import { findHandleDef } from "@/nodes/handleRegistry";
import { useEditorStore } from "@/stores/editorStore";
import connectionsData from "@/data/connections.json";

/** Cross-category connection matrix from connections.json */
const connectionMatrix = connectionsData.connectionMatrix as Record<string, Record<string, number>>;

/**
 * Check if a cross-category connection is valid according to the connection matrix.
 * The matrix uses category names from the V2 schema (e.g. "Density", "Curve").
 */
function isCrossCategoryAllowed(sourceCategory: string, targetCategory: string): boolean {
  return (connectionMatrix[sourceCategory]?.[targetCategory] ?? 0) > 0;
}

/**
 * Connection type validation for ReactFlow edges.
 * Ensures source/target handle categories match, or that the
 * cross-category connection is allowed by the connection matrix.
 */
export function useConnectionValidation() {
  return useCallback(
    (connection: Edge | Connection) => {
      try {
        if (!findHandleDef) return true;

        const source = connection.source;
        const target = connection.target;
        const sourceHandle = connection.sourceHandle;
        const targetHandle = connection.targetHandle;

        if (!source || !target || !sourceHandle || !targetHandle) return true;

        const nodes = useEditorStore.getState().nodes;
        const sourceNode = nodes.find((n) => n.id === source);
        const targetNode = nodes.find((n) => n.id === target);
        if (!sourceNode || !targetNode) return true;

        const sourceType = sourceNode.type ?? "default";
        const targetType = targetNode.type ?? "default";

        // GenericNode / group / Root fallback: allow all connections
        if (sourceType === "default" || targetType === "default") return true;
        if (sourceType === "group" || targetType === "group") return true;
        if (sourceType === "Root" || targetType === "Root") return true;

        const sourceDef = findHandleDef(sourceType, sourceHandle);
        const targetDef = findHandleDef(targetType, targetHandle);

        // If either handle isn't in the registry, allow (unknown/dynamic handles)
        if (!sourceDef || !targetDef) return true;

        // Same-category: always valid
        if (sourceDef.category === targetDef.category) return true;

        // Cross-category: check the connection matrix
        return isCrossCategoryAllowed(sourceDef.category, targetDef.category);
      } catch (err) {
        if (import.meta.env.DEV) console.warn("isValidConnection error:", err);
        return true;
      }
    },
    [],
  );
}

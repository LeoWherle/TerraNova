import { describe, it, expect } from "vitest";
import { graphToJson } from "../graphToJson";
import { jsonToGraph } from "../jsonToGraph";

describe("round-trip: jsonToGraph → graphToJson", () => {
  it("preserves scalar fields for a single node", () => {
    const original = {
      Type: "SimplexNoise2D",
      Frequency: 0.01,
      Amplitude: 1.5,
      Seed: 42,
    };

    const { nodes, edges } = jsonToGraph(original);
    const result = graphToJson(nodes, edges);

    expect(result).toEqual(original);
  });

  it("preserves nested structure for two connected nodes", () => {
    const original = {
      Type: "Clamp",
      Min: -1,
      Max: 1,
      Input: {
        Type: "SimplexNoise2D",
        Frequency: 0.05,
        Amplitude: 2,
      },
    };

    const { nodes, edges } = jsonToGraph(original);
    const result = graphToJson(nodes, edges);

    expect(result).toEqual(original);
  });

  it("preserves deeply nested chains", () => {
    const original = {
      Type: "Negate",
      Input: {
        Type: "Clamp",
        Min: 0,
        Max: 1,
        Input: {
          Type: "Constant",
          Value: 0.75,
        },
      },
    };

    const { nodes, edges } = jsonToGraph(original);
    const result = graphToJson(nodes, edges);

    expect(result).toEqual(original);
  });

  it("preserves cross-category references", () => {
    const original = {
      Type: "CurveFunction",
      Curve: {
        Type: "Manual",
        Points: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
      },
      Input: {
        Type: "Constant",
        Value: 0.5,
      },
    };

    const { nodes, edges } = jsonToGraph(original);
    const result = graphToJson(nodes, edges);

    expect(result).toEqual(original);
  });

  it("preserves boolean and string fields", () => {
    const original = {
      Type: "SimplexNoise2D",
      Frequency: 0.01,
      Amplitude: 1,
      Skip: true,
      ExportAs: "myNoise",
    };

    const { nodes, edges } = jsonToGraph(original);
    const result = graphToJson(nodes, edges);

    expect(result).toEqual(original);
  });

  it("preserves array-of-assets fields", () => {
    // Sum now uses Inputs[] natively via compound handles.
    // InputA/InputB in source JSON gets migrated to Inputs[0]/Inputs[1] by jsonToGraph,
    // and graphToJson serializes them as Inputs: [...] array.
    const original = {
      Type: "Sum",
      InputA: { Type: "Constant", Value: 1 },
      InputB: { Type: "Constant", Value: 2 },
    };

    const { nodes, edges } = jsonToGraph(original);
    const result = graphToJson(nodes, edges);

    // After migration, the round-trip produces Inputs[] format
    expect(result).toEqual({
      Type: "Sum",
      Inputs: [
        { Type: "Constant", Value: 1 },
        { Type: "Constant", Value: 2 },
      ],
    });
  });

  it("preserves mixed scalar and array-of-assets fields", () => {
    const original = {
      Type: "WeightedSum",
      Weight: 0.5,
      Inputs: [
        { Type: "Constant", Value: 10 },
        { Type: "Constant", Value: 20 },
      ],
      Input: {
        Type: "SimplexNoise2D",
        Frequency: 0.01,
      },
    };

    const { nodes, edges } = jsonToGraph(original);
    const result = graphToJson(nodes, edges);

    expect(result).toEqual(original);
  });

  it("preserves nested assets in fields not in NESTED_ASSET_FIELDS allowlist", () => {
    const original = {
      Type: "CustomFilter",
      Threshold: 0.5,
      FooBarInput: {
        Type: "SimplexNoise2D",
        Frequency: 0.03,
        Amplitude: 1,
      },
    };

    const { nodes, edges } = jsonToGraph(original);
    expect(nodes).toHaveLength(2);
    expect(edges).toHaveLength(1);

    const result = graphToJson(nodes, edges);
    expect(result).toEqual(original);
  });

  it("handles a 5-node chain round-trip", () => {
    const original = {
      Type: "Negate",
      Input: {
        Type: "Abs",
        Input: {
          Type: "LinearTransform",
          Scale: 2,
          Offset: -0.5,
          Input: {
            Type: "Clamp",
            Min: 0,
            Max: 1,
            Input: {
              Type: "SimplexNoise2D",
              Frequency: 0.02,
              Amplitude: 1,
            },
          },
        },
      },
    };

    const { nodes, edges } = jsonToGraph(original);
    expect(nodes).toHaveLength(5);
    expect(edges).toHaveLength(4);

    const result = graphToJson(nodes, edges);
    expect(result).toEqual(original);
  });

  it("preserves Vector:Constant with object Value through graph round-trip", () => {
    const original = {
      Type: "Box",
      Range: {
        Type: "Constant",
        Value: { x: 3, y: 5, z: 3 },
      },
    };

    const { nodes, edges } = jsonToGraph(original);
    expect(nodes).toHaveLength(2);
    expect(edges).toHaveLength(1);

    // The child node should be typed as Vector:Constant
    const childNode = nodes.find((n) => n.type === "Vector:Constant");
    expect(childNode).toBeDefined();
    expect((childNode!.data as Record<string, unknown>).type).toBe("Constant");
    const fields = (childNode!.data as Record<string, unknown>).fields as Record<string, unknown>;
    expect(fields.Value).toEqual({ x: 3, y: 5, z: 3 });

    const result = graphToJson(nodes, edges);
    expect(result).toEqual(original);
  });
});

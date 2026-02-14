import { create } from "zustand";

interface DragData {
  nodeType: string;
  displayType: string;
  defaults: Record<string, unknown>;
}

interface DragState {
  isDragging: boolean;
  dragData: DragData | null;
  cursorPos: { x: number; y: number };

  /** Category of the handle currently being dragged during a connection attempt */
  connectingCategory: string | null;

  startDrag: (data: DragData) => void;
  updateCursor: (x: number, y: number) => void;
  endDrag: () => void;
  setConnectingCategory: (cat: string | null) => void;
}

export const useDragStore = create<DragState>((set) => ({
  isDragging: false,
  dragData: null,
  cursorPos: { x: 0, y: 0 },
  connectingCategory: null,

  startDrag: (dragData) => set({ isDragging: true, dragData }),
  updateCursor: (x, y) => set({ cursorPos: { x, y } }),
  endDrag: () => set({ isDragging: false, dragData: null }),
  setConnectingCategory: (connectingCategory) => set({ connectingCategory }),
}));

import { getCurrentWindow } from "@tauri-apps/api/window";
import { useEffect, useState } from "react";
import { isMac } from "@/utils/platform";

// Simple title bar for home screen (no menus, no ReactFlow dependencies)
export function SimpleTitleBar() {
  const [isMaximized, setIsMaximized] = useState(false);
  const appWindow = getCurrentWindow();

  useEffect(() => {
    if (isMac) return;

    appWindow.isMaximized().then(setIsMaximized);

    const unlisten = appWindow.onResized(() => {
      appWindow.isMaximized().then(setIsMaximized);
    });

    return () => {
      unlisten.then((fn) => fn());
    };
  }, [appWindow]);

  const handleMinimize = () => {
    appWindow.minimize();
  };

  const handleMaximize = () => {
    appWindow.toggleMaximize();
  };

  const handleClose = () => {
    appWindow.close();
  };

  // On macOS, native traffic lights handle window controls — just render a minimal drag region
  if (isMac) {
    return (
      <div className="h-8 bg-tn-panel border-b border-tn-border flex items-center select-none shrink-0" />
    );
  }

  return (
    <div
      data-tauri-drag-region
      className="h-8 bg-tn-panel border-b border-tn-border flex items-center justify-between select-none shrink-0"
    >
      <div data-tauri-drag-region className="flex-1" />

      <div className="flex items-center gap-0">
        <button
          onClick={handleMinimize}
          className="w-12 h-8 flex items-center justify-center hover:bg-tn-surface transition-colors text-tn-text-muted hover:text-tn-text"
          title="Minimize"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <line x1="5" y1="12" x2="19" y2="12" />
          </svg>
        </button>

        <button
          onClick={handleMaximize}
          className="w-12 h-8 flex items-center justify-center hover:bg-tn-surface transition-colors text-tn-text-muted hover:text-tn-text"
          title={isMaximized ? "Restore" : "Maximize"}
        >
          {isMaximized ? (
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
              <line x1="9" y1="3" x2="9" y2="21" />
            </svg>
          ) : (
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
            </svg>
          )}
        </button>

        <button
          onClick={handleClose}
          className="w-12 h-8 flex items-center justify-center hover:bg-red-600 transition-colors text-tn-text-muted hover:text-white"
          title="Close"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <line x1="18" y1="6" x2="6" y2="18" />
            <line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        </button>
      </div>
    </div>
  );
}

// Export as TitleBar for backward compatibility
export { SimpleTitleBar as TitleBar };

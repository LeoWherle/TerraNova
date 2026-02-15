import { check, type Update } from "@tauri-apps/plugin-updater";
import { invoke } from "@tauri-apps/api/core";
import { useUpdateStore } from "@/stores/updateStore";
import { useToastStore } from "@/stores/toastStore";

let pendingUpdate: Update | null = null;

export async function checkForUpdates(manual = false): Promise<void> {
  const store = useUpdateStore.getState();
  if (store.status !== "idle") return;

  store.setStatus("checking");
  try {
    const update = await check();
    if (update) {
      pendingUpdate = update;
      store.setVersion(update.version);
      store.setStatus("available");
      useToastStore.getState().addToast(
        `Update v${update.version} available`,
        "info",
      );
    } else {
      store.setStatus("idle");
      if (manual) {
        useToastStore.getState().addToast("You're on the latest version", "success");
      }
    }
  } catch {
    store.setStatus("idle");
    if (manual) {
      useToastStore.getState().addToast("Could not check for updates", "error");
    }
  }
}

export async function downloadAndInstall(): Promise<void> {
  if (!pendingUpdate) return;

  const store = useUpdateStore.getState();
  const targetVersion = pendingUpdate.version;
  store.setStatus("downloading");
  store.setProgress(0);

  try {
    let contentLength = 0;
    let downloaded = 0;

    await pendingUpdate.downloadAndInstall((event) => {
      if (event.event === "Started") {
        contentLength = event.data.contentLength ?? 0;
      } else if (event.event === "Progress") {
        downloaded += event.data.chunkLength;
        if (contentLength > 0) {
          useUpdateStore.getState().setProgress(
            Math.round((downloaded / contentLength) * 100),
          );
        }
      } else if (event.event === "Finished") {
        useUpdateStore.getState().setProgress(100);
      }
    });

    pendingUpdate = null;
    useUpdateStore.getState().setStatus("restarting");

    // Record target version so the next launch can verify the update applied
    localStorage.setItem("tn-update-target", targetVersion);

    // Brief delay for filesystem flush, then relaunch.
    // Don't await — relaunch_app calls app.exit(0) which kills the process,
    // so the promise will never resolve. The catch handles the unlikely case
    // where the command fails before the process exits.
    await new Promise((resolve) => setTimeout(resolve, 500));
    invoke("relaunch_app").catch((err) => {
      console.error("Relaunch failed:", err);
      useUpdateStore.getState().setStatus("ready");
      useToastStore.getState().addToast(
        "Restart failed — please close and reopen the app",
        "error",
      );
    });
  } catch (err) {
    console.error("Update download/install failed:", err);
    useUpdateStore.getState().setError(String(err));
    useUpdateStore.getState().setStatus("idle");
    useToastStore.getState().addToast("Update download failed", "error");
    pendingUpdate = null;
  }
}

export async function restartToUpdate(): Promise<void> {
  useUpdateStore.getState().setStatus("restarting");

  // Record target version for post-update verification
  const targetVersion = useUpdateStore.getState().version;
  if (targetVersion) {
    localStorage.setItem("tn-update-target", targetVersion);
  }

  // Fire-and-forget — the process will exit
  invoke("relaunch_app").catch((err) => {
    console.error("Relaunch failed:", err);
    useUpdateStore.getState().setStatus("ready");
    useToastStore.getState().addToast(
      "Restart failed — please close and reopen the app",
      "error",
    );
  });
}

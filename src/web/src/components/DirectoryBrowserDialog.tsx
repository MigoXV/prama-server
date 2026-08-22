import { Folder, FolderOpen, RefreshCw, Volume2, X } from "lucide-react";
import { useCallback, useEffect, useId, useRef, useState } from "react";
import { createPortal } from "react-dom";
import type { ServerDirectoryListing } from "../types";
import { Button } from "./ui";

interface DirectoryBrowserDialogProps {
  isOpen: boolean;
  initialPath?: string;
  listDirectory: (path?: string) => Promise<ServerDirectoryListing>;
  onSelect: (path: string) => void;
  onClose: () => void;
}

const FOCUSABLE_SELECTOR = [
  "button:not([disabled])",
  "input:not([disabled])",
  "select:not([disabled])",
  "textarea:not([disabled])",
  "a[href]",
  '[tabindex]:not([tabindex="-1"])',
].join(",");

export function DirectoryBrowserDialog({
  isOpen,
  initialPath,
  listDirectory,
  onSelect,
  onClose,
}: DirectoryBrowserDialogProps) {
  const [listing, setListing] = useState<ServerDirectoryListing | null>(null);
  const [requestedPath, setRequestedPath] = useState(initialPath ?? "");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const titleId = useId();
  const descriptionId = useId();
  const dialogRef = useRef<HTMLDivElement | null>(null);
  const closeButtonRef = useRef<HTMLButtonElement | null>(null);
  const requestIdRef = useRef(0);
  const returnFocusRef = useRef<HTMLElement | null>(null);

  const loadDirectory = useCallback(
    async (path?: string) => {
      const requestId = requestIdRef.current + 1;
      requestIdRef.current = requestId;
      setLoading(true);
      setError("");
      try {
        const nextListing = await listDirectory(path);
        if (requestId !== requestIdRef.current) {
          return;
        }
        setListing(nextListing);
        setRequestedPath(nextListing.currentPath);
      } catch (reason) {
        if (requestId === requestIdRef.current) {
          setError(reason instanceof Error ? reason.message : "目录读取失败");
        }
      } finally {
        if (requestId === requestIdRef.current) {
          setLoading(false);
        }
      }
    },
    [listDirectory],
  );

  useEffect(() => {
    if (!isOpen) {
      return;
    }
    returnFocusRef.current = document.activeElement as HTMLElement | null;
    setListing(null);
    setRequestedPath(initialPath ?? "");
    void loadDirectory(initialPath || undefined);
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    window.requestAnimationFrame(() => closeButtonRef.current?.focus());
    return () => {
      requestIdRef.current += 1;
      document.body.style.overflow = previousOverflow;
      returnFocusRef.current?.focus();
    };
  }, [initialPath, isOpen, loadDirectory]);

  useEffect(() => {
    if (!isOpen) {
      return;
    }
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        event.preventDefault();
        onClose();
        return;
      }
      if (event.key !== "Tab" || !dialogRef.current) {
        return;
      }
      const focusable = Array.from(
        dialogRef.current.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR),
      );
      if (!focusable.length) {
        event.preventDefault();
        return;
      }
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) {
    return null;
  }

  const currentPath = listing?.currentPath ?? requestedPath;

  return createPortal(
    <div className="dialog-backdrop" onMouseDown={(event) => {
      if (event.target === event.currentTarget) {
        onClose();
      }
    }}>
      <div
        ref={dialogRef}
        className="dialog directory-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-describedby={descriptionId}
      >
        <header className="dialog-header">
          <div>
            <p className="section-kicker">目录浏览器</p>
            <h2 id={titleId}>选择服务器目录</h2>
          </div>
          <button
            ref={closeButtonRef}
            type="button"
            className="icon-action"
            aria-label="关闭目录浏览器"
            onClick={onClose}
          >
            <X size={18} />
          </button>
        </header>

        <div className="directory-toolbar">
          <div className="directory-current-path">
            <span>当前位置</span>
            <code>{currentPath || "正在读取"}</code>
          </div>
          <Button
            variant="ghost"
            disabled={loading || !currentPath}
            onClick={() => void loadDirectory(currentPath || undefined)}
          >
            <RefreshCw size={15} aria-hidden="true" />
            刷新
          </Button>
        </div>

        {error ? (
          <div className="dialog-error" role="alert">
            <span>{error}</span>
            <Button variant="ghost" onClick={() => void loadDirectory(currentPath || undefined)}>
              重新读取
            </Button>
          </div>
        ) : null}

        <div className="directory-list" aria-busy={loading}>
          {listing?.parentPath ? (
            <button
              type="button"
              className="directory-row"
              disabled={loading}
              onClick={() => void loadDirectory(listing.parentPath ?? undefined)}
            >
              <FolderOpen size={18} aria-hidden="true" />
              <span>上级目录</span>
            </button>
          ) : null}
          {loading && !listing ? (
            <div className="directory-empty" role="status">正在读取目录…</div>
          ) : null}
          {!loading && listing && listing.entries.length === 0 ? (
            <div className="directory-empty">当前目录为空</div>
          ) : null}
          {listing?.entries.map((entry) =>
            entry.kind === "directory" ? (
              <button
                type="button"
                className="directory-row"
                key={entry.path}
                disabled={loading}
                onClick={() => void loadDirectory(entry.path)}
              >
                <Folder size={18} aria-hidden="true" />
                <span>{entry.name}</span>
              </button>
            ) : (
              <div className="directory-row directory-file" key={entry.path}>
                <Volume2 size={18} aria-hidden="true" />
                <span>{entry.name}</span>
              </div>
            ),
          )}
        </div>

        <footer className="dialog-footer">
          <p id={descriptionId}>选择后将扫描当前目录中的音频文件。</p>
          <Button
            disabled={!currentPath || loading}
            onClick={() => onSelect(currentPath)}
          >
            选择当前目录
          </Button>
        </footer>
      </div>
    </div>,
    document.body,
  );
}

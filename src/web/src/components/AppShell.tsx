import {
  Activity,
  BookOpen,
  Menu,
  PanelLeftClose,
  PanelLeftOpen,
  Plus,
  Settings,
  X,
} from "lucide-react";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { usePersistentState } from "../hooks/usePersistentState";
import type { AppRoute } from "../hooks/useRoute";

export function AppShell({
  page,
  navigate,
  children,
}: {
  page: AppRoute["page"];
  navigate: (url: string) => void;
  children: ReactNode;
}) {
  const [collapsed, setCollapsed] = usePersistentState(
    "prama.navigationCollapsed",
    false,
  );
  const [open, setOpen] = useState(false);
  const dialog = useRef<HTMLDialogElement>(null);
  const menu = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    if (open) dialog.current?.showModal();
    else if (dialog.current?.open) dialog.current.close();
  }, [open]);
  useEffect(() => {
    const mq = matchMedia("(min-width: 641px)");
    const resize = () => {
      if (mq.matches) setOpen(false);
    };
    mq.addEventListener("change", resize);
    return () => mq.removeEventListener("change", resize);
  }, []);
  const items = [
    { label: "新建评估", path: "/evaluations/new", id: "new", icon: Plus },
    { label: "当前任务", path: "/evaluations", id: "tasks", icon: Activity },
    { label: "默认设置", path: "/settings", id: "settings", icon: Settings },
    { label: "帮助", path: "/help", id: "help", icon: BookOpen },
  ];
  function nav(mobile = false) {
    return (
      <>
        <div className="paper-brand">
          <span>
            Prama<span className="brand-period">.</span>
          </span>
          {mobile ? (
            <button
              className="icon-button"
              aria-label="关闭导航"
              onClick={() => setOpen(false)}
            >
              <X size={18} />
            </button>
          ) : (
            <button
              className="icon-button collapse-control"
              aria-label={collapsed ? "展开导航" : "收起导航"}
              aria-expanded={!collapsed}
              onClick={() => setCollapsed(!collapsed)}
            >
              {collapsed ? (
                <PanelLeftOpen size={17} />
              ) : (
                <PanelLeftClose size={17} />
              )}
            </button>
          )}
        </div>
        <p className="workspace-name">语音评估工作空间</p>
        <nav aria-label="主导航">
          {items.map(({ label, path, id, icon: Icon }) => (
            <a
              key={id}
              href={path}
              title={label}
              aria-label={label}
              aria-current={
                page === id || (id === "tasks" && page === "job")
                  ? "page"
                  : undefined
              }
              onClick={(e) => {
                if (e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
                e.preventDefault();
                navigate(path);
                setOpen(false);
              }}
            >
              <Icon size={17} aria-hidden />
              <span>{label}</span>
            </a>
          ))}
        </nav>
        <div className="sidebar-foot">
          <span className="small-dot" />
          评估控制台
          <span className="sidebar-foot-note">
            ASR · VAD · LID · Keyword · SE
          </span>
        </div>
      </>
    );
  }
  return (
    <div className={`paper-app ${collapsed ? "navigation-collapsed" : ""}`}>
      <a className="skip-link" href="#main-content">
        跳到主内容
      </a>
      <aside className="paper-sidebar">{nav()}</aside>
      <div className="paper-workspace">
        <div className="paper-topbar">
          <button
            ref={menu}
            className="icon-button mobile-menu"
            aria-label="打开导航"
            aria-expanded={open}
            onClick={() => setOpen(true)}
          >
            <Menu size={18} />
          </button>
          <span>
            Prama <span className="breadcrumb-divider">/</span>{" "}
            {page === "new"
              ? "新建评估"
              : page === "settings"
                ? "默认设置"
                : page === "help"
                  ? "帮助"
                  : "评估任务"}
          </span>
          <span className="topbar-note">语音评估工作空间</span>
        </div>
        <main id="main-content" tabIndex={-1} className="paper-main" key={page}>
          {children}
        </main>
      </div>
      <dialog
        ref={dialog}
        className="mobile-navigation"
        aria-label="导航"
        onCancel={() => setOpen(false)}
        onClose={() => {
          setOpen(false);
          menu.current?.focus();
        }}
      >
        {nav(true)}
      </dialog>
    </div>
  );
}

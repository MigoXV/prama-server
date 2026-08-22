import { forwardRef } from "react";
import type {
  ButtonHTMLAttributes,
  HTMLAttributes,
  LabelHTMLAttributes,
  ReactNode,
} from "react";

function joinClasses(...classes: Array<string | false | null | undefined>) {
  return classes.filter(Boolean).join(" ");
}

type ButtonVariant = "primary" | "secondary" | "ghost" | "danger";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  stretch?: boolean;
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  {
    variant = "primary",
    stretch = false,
    className,
    type = "button",
    ...props
  },
  ref,
) {
  return (
    <button
      ref={ref}
      type={type}
      className={joinClasses(
        "ui-button",
        `ui-button-${variant}`,
        stretch && "ui-button-stretch",
        className,
      )}
      {...props}
    />
  );
});

export function GhostButton(props: ButtonProps) {
  return <Button variant="ghost" {...props} />;
}

interface FieldProps extends LabelHTMLAttributes<HTMLLabelElement> {
  label: ReactNode;
  hint?: ReactNode;
  error?: ReactNode;
}

export function Field({ label, hint, error, className, children, ...props }: FieldProps) {
  return (
    <label className={joinClasses("ui-field", className)} {...props}>
      <span className="ui-field-label">{label}</span>
      {children}
      {error ? <span className="ui-field-error">{error}</span> : null}
      {!error && hint ? <span className="ui-field-hint">{hint}</span> : null}
    </label>
  );
}

interface MetricTileProps extends HTMLAttributes<HTMLDivElement> {
  label: ReactNode;
  value: ReactNode;
  detail?: ReactNode;
}

export function MetricTile({ label, value, detail, className, ...props }: MetricTileProps) {
  return (
    <div className={joinClasses("ui-metric", className)} {...props}>
      <span className="ui-metric-label">{label}</span>
      <strong className="ui-metric-value">{value}</strong>
      {detail ? <small className="ui-metric-detail">{detail}</small> : null}
    </div>
  );
}

export function StatusChip({ className, ...props }: HTMLAttributes<HTMLSpanElement>) {
  return <span className={joinClasses("ui-status", className)} {...props} />;
}

export function WorkbenchShell({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return <div className={joinClasses("app-shell", className)} {...props} />;
}

export function SidebarPane({ className, ...props }: HTMLAttributes<HTMLElement>) {
  return <aside className={joinClasses("app-sidebar", className)} {...props} />;
}

export function WorkspacePane({ className, ...props }: HTMLAttributes<HTMLElement>) {
  return <main id="main-content" className={joinClasses("app-workspace", className)} {...props} />;
}

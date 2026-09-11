import { cn } from "@/lib/utils";

/**
 * The error banner. It was the same class string written three times across
 * two components, already divergent in text size.
 */
export function ErrorBanner({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <p
      role="alert"
      className={cn(
        "rounded-md border border-red-500/40 bg-red-500/10 px-3 py-2 text-red-500",
        className,
      )}
    >
      {children}
    </p>
  );
}

import { LineChart } from "lucide-react";
import { Link, NavLink, Outlet } from "react-router-dom";
import { cn } from "@/lib/utils";
import { ThemeToggle } from "./ThemeToggle";
import { TickerSearch } from "./TickerSearch";

const nav = [
  { to: "/", label: "Home" },
  { to: "/build", label: "Strategy builder" },
];

export function Layout() {
  return (
    <div className="flex min-h-screen flex-col">
      <header className="sticky top-0 z-30 border-b border-border bg-bg/80 backdrop-blur">
        <div className="mx-auto flex h-14 max-w-7xl items-center gap-4 px-4 sm:px-6">
          <Link to="/" className="flex items-center gap-2 font-semibold tracking-tight">
            <span className="grid h-7 w-7 place-items-center rounded-lg bg-accent text-white">
              <LineChart className="h-4 w-4" />
            </span>
            <span className="hidden sm:inline">Beat The Market</span>
          </Link>
          <nav className="flex items-center gap-1">
            {nav.map((n) => (
              <NavLink
                key={n.to}
                to={n.to}
                end
                className={({ isActive }) =>
                  cn(
                    "rounded-md px-3 py-1.5 text-sm transition-colors",
                    isActive ? "bg-surface-2 text-text" : "text-muted hover:text-text",
                  )
                }
              >
                {n.label}
              </NavLink>
            ))}
          </nav>
          <TickerSearch className="ml-auto w-full max-w-xs" />
          <ThemeToggle />
        </div>
      </header>
      <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-6 sm:px-6">
        <Outlet />
      </main>
      <footer className="border-t border-border py-4 text-center text-xs text-muted">
        Research tool, not financial advice. Past results don't guarantee future returns.
      </footer>
    </div>
  );
}

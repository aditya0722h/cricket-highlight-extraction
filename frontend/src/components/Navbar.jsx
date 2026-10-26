import React from 'react';

export const Navbar = () => {
  return (
    <header className="border-b border-[var(--card-border)] bg-[var(--card-bg)]/80 backdrop-blur-xl sticky top-0 z-50 transition-colors duration-200">
      <div className="max-w-7xl mx-auto px-6 h-20 flex items-center justify-between">
        <div className="flex items-center space-x-3.5">
          <div className="w-10 h-10 rounded-2xl bg-stone-200 dark:bg-stone-800 border border-[var(--card-border)] flex items-center justify-center text-lg">
            🏏
          </div>
          <div>
            <h1 className="text-xl font-bold font-display tracking-tight">
              Cricket Highlights
            </h1>
            <span className="text-[var(--text-muted)] text-xs font-medium tracking-widest uppercase block">
              AI Engine
            </span>
          </div>
        </div>
        <div className="flex items-center space-x-2 text-xs font-medium text-[var(--text-muted)] bg-[var(--bg-primary)] border border-[var(--card-border)] px-4 py-2 rounded-full shadow-sm">
          <span className="w-2 h-2 rounded-full bg-emerald-600 dark:bg-emerald-400"></span>
          <span>Backend Connected</span>
        </div>
      </div>
    </header>
  );
};
import React from 'react';

export const StatusBadge = ({ status }) => {
  const styles = {
    completed: 'bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 border-emerald-500/20',
    processing: 'bg-stone-500/10 text-stone-700 dark:text-stone-300 border-stone-500/30 animate-pulse',
    queued: 'bg-amber-500/10 text-amber-700 dark:text-amber-400 border-amber-500/20',
    failed: 'bg-rose-500/10 text-rose-700 dark:text-rose-400 border-rose-500/20',
  };

  return (
    <span className={`inline-flex items-center gap-1.5 px-3 py-1 text-xs font-medium rounded-full border tracking-wide uppercase ${styles[status] || styles.queued}`}>
      {status === 'processing' && <span className="w-1.5 h-1.5 rounded-full bg-stone-500 dark:bg-stone-400 animate-ping"></span>}
      {status}
    </span>
  );
};
import { StatusBadge } from './StatusBadge.jsx'

export function JobCard({ job }) {
  return (
    <article className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">
            {job.sport} · {job.jobType} job {job.id ? `#${job.id}` : ''}
          </p>
          <h3 className="mt-2 break-all text-base font-semibold text-slate-900">
            {job.videoPath}
          </h3>
        </div>
        <StatusBadge status={job.status} />
      </div>

      <div className="mt-5">
        <p className="mb-3 text-sm text-slate-600">
          <span className="font-medium text-slate-700">Export:</span>{' '}
          {job.aspect} · {job.editStyle}
        </p>
        <div className="mb-2 flex items-center justify-between text-sm">
          <span className="font-medium text-slate-600">Progress</span>
          <span className="font-mono font-semibold text-slate-900">
            {job.progress}%
          </span>
        </div>
        <div
          className="h-2.5 overflow-hidden rounded-full bg-slate-100"
          role="progressbar"
          aria-label={`Job progress: ${job.progress}%`}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={job.progress}
        >
          <div
            className={`h-full rounded-full transition-[width] duration-500 ${
              job.status === 'failed' ? 'bg-rose-500' : 'bg-emerald-500'
            }`}
            style={{ width: `${job.progress}%` }}
          />
        </div>
      </div>

      {job.error && (
        <p className="mt-4 rounded-xl border border-rose-200 bg-rose-50 px-3 py-2 text-sm text-rose-800">
          {job.error}
        </p>
      )}
      {!job.error && job.message && (
        <p className="mt-4 text-sm text-slate-600">{job.message}</p>
      )}
      {job.outputFiles.length > 0 && (
        <div className="mt-4 border-t border-slate-100 pt-4">
          <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
            Download outputs
          </p>
          <ul className="space-y-2">
            {job.outputFiles.map((file) => (
              <li key={file.path}>
                <a
                  className="text-sm font-medium text-emerald-700 underline decoration-emerald-300 underline-offset-2 hover:text-emerald-900"
                  href={file.url}
                  download={file.name}
                >
                  {file.name}
                </a>
                <span className="ml-2 break-all text-xs text-slate-500">
                  {file.path}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </article>
  )
}

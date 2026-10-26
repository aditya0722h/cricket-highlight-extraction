import { useCallback, useEffect, useRef, useState } from 'react'
import { JobCard } from '../components/JobCard.jsx'
import { fetchJobs, startLiveJob, startReplayJob } from '../services/api.js'

function firstValue(record, fields, fallback = '') {
  for (const field of fields) {
    if (record[field] !== undefined && record[field] !== null) {
      return record[field]
    }
  }
  return fallback
}

function mapJob(record) {
  if (!record || typeof record !== 'object' || Array.isArray(record)) {
    throw new Error('The jobs response contains an invalid job record.')
  }

  const progressValue = Number.parseFloat(
    firstValue(record, ['progress', 'progress_percent', 'percent'], 0),
  )
  const error = firstValue(
    record,
    ['error_message', 'error', 'error_log'],
    '',
  )

  return {
    id: firstValue(record, ['id', 'job_id', 'uuid']),
    jobType: String(firstValue(record, ['job_type', 'type'], 'replay')),
    videoPath: String(
      firstValue(
        record,
        ['source_path', 'video_path', 'file_path', 'stream_url', 'url'],
        'Unknown video',
      ),
    ),
    status: String(firstValue(record, ['status', 'state'], 'queued')).toLowerCase(),
    progress: Number.isFinite(progressValue)
      ? Math.min(100, Math.max(0, Math.round(progressValue)))
      : 0,
    error: typeof error === 'string' ? error : JSON.stringify(error),
    message: String(
      firstValue(record, ['progress_message', 'message'], ''),
    ),
    aspect: String(firstValue(record, ['aspect'], 'landscape')),
    editStyle: String(firstValue(record, ['edit_style', 'editStyle'], 'plain')),
    sport: String(firstValue(record, ['sport'], 'cricket')),
    outputFiles: Array.isArray(record.output_files)
      ? record.output_files
      : [],
  }
}

function mapJobsResponse(payload) {
  const records = Array.isArray(payload) ? payload : payload?.jobs
  if (!Array.isArray(records)) {
    throw new Error('The jobs response was not a list.')
  }
  return records.map(mapJob)
}

export function Dashboard() {
  const [jobs, setJobs] = useState([])
  const [videoPath, setVideoPath] = useState('match.mp4')
  const [replayOutputDir, setReplayOutputDir] = useState('./output_highlights')
  const [replayAspect, setReplayAspect] = useState('landscape')
  const [replayEditStyle, setReplayEditStyle] = useState('plain')
  const [replaySport, setReplaySport] = useState('cricket')
  const [streamUrl, setStreamUrl] = useState('')
  const [liveOutputDir, setLiveOutputDir] = useState('./output_highlights')
  const [liveAspect, setLiveAspect] = useState('landscape')
  const [liveEditStyle, setLiveEditStyle] = useState('plain')
  const [liveSport, setLiveSport] = useState('cricket')
  const [loading, setLoading] = useState(true)
  const [submitting, setSubmitting] = useState(false)
  const [backendUnavailable, setBackendUnavailable] = useState(null)
  const [submissionError, setSubmissionError] = useState('')
  const requestInFlight = useRef(false)

  const refreshJobs = useCallback(async () => {
    if (requestInFlight.current) return
    requestInFlight.current = true

    try {
      setJobs(mapJobsResponse(await fetchJobs()))
      setBackendUnavailable(false)
    } catch {
      setBackendUnavailable(true)
    } finally {
      requestInFlight.current = false
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    const initialTimeoutId = window.setTimeout(refreshJobs, 0)
    const intervalId = window.setInterval(refreshJobs, 2000)
    return () => {
      window.clearTimeout(initialTimeoutId)
      window.clearInterval(intervalId)
    }
  }, [refreshJobs])

  const handleReplaySubmit = async (event) => {
    event.preventDefault()
    setSubmitting(true)
    setSubmissionError('')

    try {
      await startReplayJob(
        videoPath.trim(),
        replayOutputDir.trim(),
        replayAspect,
        replayEditStyle,
        replaySport,
      )
      await refreshJobs()
    } catch (error) {
      setSubmissionError(error.message || 'Could not queue the replay job.')
    } finally {
      setSubmitting(false)
    }
  }

  const handleLiveSubmit = async (event) => {
    event.preventDefault()
    setSubmitting(true)
    setSubmissionError('')

    try {
      await startLiveJob(
        streamUrl.trim(),
        liveOutputDir.trim(),
        liveAspect,
        liveEditStyle,
        liveSport,
      )
      await refreshJobs()
    } catch (error) {
      setSubmissionError(error.message || 'Could not queue the live job.')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <main className="min-h-screen bg-slate-50 text-slate-900">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-4 px-5 py-5 sm:px-8">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-emerald-700">
              AI Engine
            </p>
            <h1 className="mt-1 text-2xl font-bold tracking-tight">
              Match Highlights
            </h1>
          </div>
          <div
            className={`inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-sm ${
              backendUnavailable === null
                ? 'border-slate-200 bg-slate-50 text-slate-600'
                : backendUnavailable
                ? 'border-rose-200 bg-rose-50 text-rose-700'
                : 'border-emerald-200 bg-emerald-50 text-emerald-800'
            }`}
            role="status"
          >
            <span
              className={`h-2 w-2 rounded-full ${
                backendUnavailable === null
                  ? 'bg-slate-400'
                  : backendUnavailable
                    ? 'bg-rose-500'
                    : 'bg-emerald-500'
              }`}
            />
            {backendUnavailable === null
              ? 'Checking backend…'
              : backendUnavailable
                ? 'Backend unavailable'
                : 'Backend connected'}
          </div>
        </div>
      </header>

      <div className="mx-auto max-w-6xl space-y-8 px-5 py-8 sm:px-8">
        {backendUnavailable === true && (
          <div
            className="rounded-2xl border border-rose-200 bg-rose-50 px-5 py-4 text-sm text-rose-800"
            role="alert"
          >
            Cannot reach the Flask backend. Start it at{' '}
            <code className="font-mono">http://localhost:5000</code>; jobs will
            refresh automatically when it becomes available.
          </div>
        )}

        {submissionError && (
          <div
            className="rounded-2xl border border-rose-200 bg-rose-50 px-5 py-4 text-sm text-rose-800"
            role="alert"
          >
            {submissionError}
          </div>
        )}

        <section className="grid gap-5 lg:grid-cols-2" aria-label="Queue jobs">
          <form
            onSubmit={handleReplaySubmit}
            className="space-y-5 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm"
          >
            <div>
              <h2 className="text-lg font-semibold">Process a replay</h2>
              <p className="mt-1 text-sm text-slate-600">
                Queue a local match video for highlight extraction.
              </p>
            </div>
            <label className="block space-y-1.5 text-sm font-medium">
              Video path
              <input
                className="w-full rounded-xl border border-slate-300 px-3.5 py-2.5 font-mono text-sm outline-none focus:border-emerald-600 focus:ring-2 focus:ring-emerald-100"
                type="text"
                value={videoPath}
                onChange={(event) => setVideoPath(event.target.value)}
                placeholder="match.mp4"
                required
              />
            </label>
            <label className="block space-y-1.5 text-sm font-medium">
              Output directory
              <input
                className="w-full rounded-xl border border-slate-300 px-3.5 py-2.5 font-mono text-sm outline-none focus:border-emerald-600 focus:ring-2 focus:ring-emerald-100"
                type="text"
                value={replayOutputDir}
                onChange={(event) => setReplayOutputDir(event.target.value)}
                required
              />
            </label>
            <label className="block space-y-1.5 text-sm font-medium">
              Aspect ratio
              <select
                className="w-full rounded-xl border border-slate-300 bg-white px-3.5 py-2.5 text-sm outline-none focus:border-emerald-600 focus:ring-2 focus:ring-emerald-100"
                value={replayAspect}
                onChange={(event) => setReplayAspect(event.target.value)}
              >
                <option value="landscape">Landscape</option>
                <option value="portrait">Portrait</option>
                <option value="both">Both</option>
              </select>
            </label>
            <label className="block space-y-1.5 text-sm font-medium">
              Edit style
              <select
                className="w-full rounded-xl border border-slate-300 bg-white px-3.5 py-2.5 text-sm outline-none focus:border-emerald-600 focus:ring-2 focus:ring-emerald-100"
                value={replayEditStyle}
                onChange={(event) => setReplayEditStyle(event.target.value)}
              >
                <option value="plain">Plain</option>
                <option value="slowmo">Slow motion peak</option>
                <option value="overlays">Event captions</option>
              </select>
            </label>
            <label className="block space-y-1.5 text-sm font-medium">
              Sport
              <select
                className="w-full rounded-xl border border-slate-300 bg-white px-3.5 py-2.5 text-sm outline-none focus:border-emerald-600 focus:ring-2 focus:ring-emerald-100"
                value={replaySport}
                onChange={(event) => setReplaySport(event.target.value)}
              >
                <option value="cricket">Cricket</option>
                <option value="football">Football (soccer)</option>
              </select>
            </label>
            <button
              className="w-full rounded-xl bg-emerald-700 px-4 py-3 text-sm font-semibold text-white transition hover:bg-emerald-800 disabled:cursor-not-allowed disabled:opacity-60"
              type="submit"
              disabled={submitting || !videoPath.trim()}
            >
              {submitting ? 'Queueing…' : 'Queue replay job'}
            </button>
          </form>

          <form
            onSubmit={handleLiveSubmit}
            className="space-y-5 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm"
          >
            <div>
              <h2 className="text-lg font-semibold">Process a live stream</h2>
              <p className="mt-1 text-sm text-slate-600">
                Queue an HTTP(S) stream URL for highlight extraction.
              </p>
            </div>
            <label className="block space-y-1.5 text-sm font-medium">
              Stream URL
              <input
                className="w-full rounded-xl border border-slate-300 px-3.5 py-2.5 font-mono text-sm outline-none focus:border-emerald-600 focus:ring-2 focus:ring-emerald-100"
                type="url"
                value={streamUrl}
                onChange={(event) => setStreamUrl(event.target.value)}
                placeholder="https://example.com/live.m3u8"
                required
              />
            </label>
            <label className="block space-y-1.5 text-sm font-medium">
              Output directory
              <input
                className="w-full rounded-xl border border-slate-300 px-3.5 py-2.5 font-mono text-sm outline-none focus:border-emerald-600 focus:ring-2 focus:ring-emerald-100"
                type="text"
                value={liveOutputDir}
                onChange={(event) => setLiveOutputDir(event.target.value)}
                required
              />
            </label>
            <label className="block space-y-1.5 text-sm font-medium">
              Aspect ratio
              <select
                className="w-full rounded-xl border border-slate-300 bg-white px-3.5 py-2.5 text-sm outline-none focus:border-emerald-600 focus:ring-2 focus:ring-emerald-100"
                value={liveAspect}
                onChange={(event) => setLiveAspect(event.target.value)}
              >
                <option value="landscape">Landscape</option>
                <option value="portrait">Portrait</option>
                <option value="both">Both</option>
              </select>
            </label>
            <label className="block space-y-1.5 text-sm font-medium">
              Edit style
              <select
                className="w-full rounded-xl border border-slate-300 bg-white px-3.5 py-2.5 text-sm outline-none focus:border-emerald-600 focus:ring-2 focus:ring-emerald-100"
                value={liveEditStyle}
                onChange={(event) => setLiveEditStyle(event.target.value)}
              >
                <option value="plain">Plain</option>
                <option value="slowmo">Slow motion peak</option>
                <option value="overlays">Event captions</option>
              </select>
            </label>
            <label className="block space-y-1.5 text-sm font-medium">
              Sport
              <select
                className="w-full rounded-xl border border-slate-300 bg-white px-3.5 py-2.5 text-sm outline-none focus:border-emerald-600 focus:ring-2 focus:ring-emerald-100"
                value={liveSport}
                onChange={(event) => setLiveSport(event.target.value)}
              >
                <option value="cricket">Cricket</option>
                <option value="football">Football (soccer)</option>
              </select>
            </label>
            <button
              className="w-full rounded-xl border border-slate-300 bg-white px-4 py-3 text-sm font-semibold text-slate-800 transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-60"
              type="submit"
              disabled={submitting || !streamUrl.trim()}
            >
              {submitting ? 'Queueing…' : 'Queue live job'}
            </button>
          </form>
        </section>

        <section aria-labelledby="jobs-heading">
          <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
            <div>
              <h2 id="jobs-heading" className="text-xl font-bold tracking-tight">
                Recent jobs
              </h2>
              <p className="mt-1 text-sm text-slate-600">
                Automatically refreshed every 2 seconds.
              </p>
            </div>
            <button
              className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm font-medium text-slate-700 hover:bg-slate-100"
              type="button"
              onClick={refreshJobs}
            >
              Refresh now
            </button>
          </div>

          {loading ? (
            <p className="rounded-2xl border border-slate-200 bg-white p-6 text-sm text-slate-600">
              Loading jobs…
            </p>
          ) : jobs.length === 0 ? (
            <p className="rounded-2xl border border-slate-200 bg-white p-6 text-sm text-slate-600">
              No jobs yet. Queue a replay or live job to get started.
            </p>
          ) : (
            <div className="grid gap-4 md:grid-cols-2">
              {jobs.map((job, index) => (
                <JobCard
                  key={job.id ?? `${job.videoPath}-${index}`}
                  job={job}
                />
              ))}
            </div>
          )}
        </section>
      </div>
    </main>
  )
}

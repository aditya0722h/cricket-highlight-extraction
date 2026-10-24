import React, { useState, useEffect } from 'react';
import { fetchJobs, startReplayJob, startLiveJob } from '../services/api';
import { StatusBadge } from '../components/StatusBadge';
import { Navbar } from '../components/Navbar';

export const Dashboard = () => {
  const [jobs, setJobs] = useState([]);
  const [videoPath, setVideoPath] = useState('match.mp4');
  const [streamUrl, setStreamUrl] = useState('');
  const [replayOutputDir, setReplayOutputDir] = useState('./outputs');
  const [liveOutputDir, setLiveOutputDir] = useState('./outputs');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const loadJobs = async () => {
    try {
      const data = await fetchJobs();
      setJobs(data);
    } catch (err) {
      console.error('Failed to load jobs:', err);
    }
  };

  useEffect(() => {
    loadJobs();
    const interval = setInterval(loadJobs, 3000);
    return () => clearInterval(interval);
  }, []);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setVideoPath(file.name);
    }
  };

  const handleReplaySubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      await startReplayJob(videoPath, replayOutputDir);
      await loadJobs();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleLiveSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      await startLiveJob(streamUrl, liveOutputDir);
      await loadJobs();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen pb-16 transition-colors duration-200">
      <Navbar />
      
      <main className="max-w-7xl mx-auto px-6 pt-10 space-y-8">
        {error && (
          <div className="bg-rose-500/10 border border-rose-500/20 text-rose-700 dark:text-rose-300 px-4 py-3.5 rounded-2xl text-sm flex items-center gap-3 shadow-sm">
            <span>⚠️</span> {error}
          </div>
        )}

        {/* Two side-by-side cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Process Local Replay Card */}
          <div className="bg-[var(--card-bg)] border border-[var(--card-border)] rounded-3xl p-7 shadow-sm transition-all duration-200">
            <div className="flex items-center gap-3 mb-2">
              <span className="text-xl">📁</span>
              <h2 className="text-xl font-bold font-display tracking-tight">Process Local Replay</h2>
            </div>
            <p className="text-[var(--text-muted)] text-xs mb-6">Select a local match file and paste your destination folder path for output clips.</p>
            
            <form onSubmit={handleReplaySubmit} className="space-y-4">
              <div className="space-y-1.5">
                <label className="block text-[11px] font-medium text-[var(--text-muted)] uppercase tracking-wider">Video Path / File</label>
                <div className="flex items-center gap-2.5">
                  <input
                    type="text"
                    value={videoPath}
                    onChange={(e) => setVideoPath(e.target.value)}
                    className="flex-1 bg-[var(--bg-primary)] border border-[var(--card-border)] rounded-2xl px-4 py-3 text-sm font-mono focus:outline-none focus:border-stone-400 dark:focus:border-stone-500 transition"
                    placeholder="match.mp4"
                    required
                  />
                  <label className="cursor-pointer bg-[var(--bg-primary)] hover:bg-stone-200 dark:hover:bg-stone-800 text-[var(--text-primary)] border border-[var(--card-border)] px-4 py-3 rounded-2xl text-xs font-semibold uppercase tracking-wider transition flex items-center gap-1.5 shrink-0">
                    <span>📂</span> Browse
                    <input 
                      type="file" 
                      accept="video/*" 
                      className="hidden" 
                      onChange={handleFileChange} 
                    />
                  </label>
                </div>
              </div>

              <div className="space-y-1.5">
                <label className="block text-[11px] font-medium text-[var(--text-muted)] uppercase tracking-wider">Output Directory Path</label>
                <input
                  type="text"
                  value={replayOutputDir}
                  onChange={(e) => setReplayOutputDir(e.target.value)}
                  className="w-full bg-[var(--bg-primary)] border border-[var(--card-border)] rounded-2xl px-4 py-3 text-sm font-mono focus:outline-none focus:border-stone-400 dark:focus:border-stone-500 transition"
                  placeholder="Paste folder path here (e.g., ./outputs)"
                  required
                />
                <p className="text-[10px] text-[var(--text-muted)] pl-1">Paste the absolute or relative path of the folder where highlights should be saved.</p>
              </div>

              <button
                type="submit"
                disabled={loading}
                className="w-full bg-stone-900 dark:bg-stone-100 hover:bg-stone-800 dark:hover:bg-white text-white dark:text-stone-900 font-semibold py-3.5 rounded-2xl text-sm transition shadow-sm disabled:opacity-50 cursor-pointer font-display tracking-wide uppercase mt-2"
              >
                Queue Replay Job
              </button>
            </form>
          </div>

          {/* Process Live Stream Card */}
          <div className="bg-[var(--card-bg)] border border-[var(--card-border)] rounded-3xl p-7 shadow-sm transition-all duration-200">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-3">
                <span className="text-xl">📡</span>
                <h2 className="text-xl font-bold font-display tracking-tight">Process Live Stream / URL</h2>
              </div>
              <span className="inline-flex items-center gap-1.5 bg-rose-500/10 border border-rose-500/20 text-rose-600 dark:text-rose-400 text-[10px] font-semibold px-2.5 py-1 rounded-full uppercase tracking-wider">
                <span className="w-1.5 h-1.5 rounded-full bg-rose-500 animate-pulse"></span> LIVE
              </span>
            </div>
            <p className="text-[var(--text-muted)] text-xs mb-6">Ingest a video stream or YouTube URL and paste your destination folder path.</p>
            
            <form onSubmit={handleLiveSubmit} className="space-y-4">
              <div className="space-y-1.5">
                <label className="block text-[11px] font-medium text-[var(--text-muted)] uppercase tracking-wider">Stream / Video URL</label>
                <input
                  type="text"
                  value={streamUrl}
                  onChange={(e) => setStreamUrl(e.target.value)}
                  className="w-full bg-[var(--bg-primary)] border border-[var(--card-border)] rounded-2xl px-4 py-3 text-sm font-mono focus:outline-none focus:border-stone-400 dark:focus:border-stone-500 transition"
                  placeholder="https://youtu.be/... or stream URL"
                  required
                />
              </div>

              <div className="space-y-1.5">
                <label className="block text-[11px] font-medium text-[var(--text-muted)] uppercase tracking-wider">Output Directory Path</label>
                <input
                  type="text"
                  value={liveOutputDir}
                  onChange={(e) => setLiveOutputDir(e.target.value)}
                  className="w-full bg-[var(--bg-primary)] border border-[var(--card-border)] rounded-2xl px-4 py-3 text-sm font-mono focus:outline-none focus:border-stone-400 dark:focus:border-stone-500 transition"
                  placeholder="Paste folder path here (e.g., ./outputs)"
                  required
                />
                <p className="text-[10px] text-[var(--text-muted)] pl-1">Paste the absolute or relative path of the folder where highlights should be saved.</p>
              </div>

              <button
                type="submit"
                disabled={loading}
                className="w-full bg-stone-900 dark:bg-stone-100 hover:bg-stone-800 dark:hover:bg-white text-white dark:text-stone-900 font-semibold py-3.5 rounded-2xl text-sm transition shadow-sm disabled:opacity-50 cursor-pointer font-display tracking-wide uppercase mt-2"
              >
                Queue Stream Job
              </button>
            </form>
          </div>
        </div>

        {/* Active Job Queue Section */}
        <div className="bg-[var(--card-bg)] border border-[var(--card-border)] rounded-3xl p-7 shadow-sm">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-bold font-display tracking-tight">Active Job Queue</h2>
            <div className="flex items-center gap-2 text-xs text-[var(--text-muted)] bg-[var(--bg-primary)] border border-[var(--card-border)] px-3.5 py-1.5 rounded-full">
              <span className="w-2 h-2 rounded-full bg-emerald-600 dark:bg-emerald-400"></span>
              <span>Polling every 3s</span>
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="border-b border-[var(--card-border)] text-[var(--text-muted)] text-xs uppercase tracking-wider font-display">
                <tr>
                  <th className="pb-4 pl-3 font-semibold">Job ID</th>
                  <th className="pb-4 font-semibold">Type</th>
                  <th className="pb-4 font-semibold">Source / Path</th>
                  <th className="pb-4 font-semibold">Status</th>
                  <th className="pb-4 font-semibold">Progress</th>
                  <th className="pb-4 pr-3 font-semibold">Message</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[var(--card-border)]">
                {jobs.length === 0 ? (
                  <tr>
                    <td colSpan="6" className="py-16 text-center">
                      <div className="flex flex-col items-center justify-center space-y-3">
                        <div className="w-14 h-14 rounded-2xl bg-[var(--bg-primary)] border border-[var(--card-border)] flex items-center justify-center text-xl shadow-inner">
                          📋
                        </div>
                        <p className="text-[var(--text-muted)] text-xs font-medium tracking-wide">No jobs found in database.</p>
                      </div>
                    </td>
                  </tr>
                ) : (
                  jobs.map((job) => (
                    <tr key={job.id} className="hover:bg-stone-500/[0.02] transition group">
                      <td className="py-4.5 pl-3 font-mono text-xs font-semibold">#{job.id}</td>
                      <td className="py-4.5 uppercase text-xs font-bold tracking-wider">{job.job_type}</td>
                      <td className="py-4.5 font-mono text-xs text-[var(--text-muted)] max-w-xs truncate" title={job.source_path}>{job.source_path}</td>
                      <td className="py-4.5"><StatusBadge status={job.status} /></td>
                      <td className="py-4.5 w-40 pr-4">
                        <div className="flex items-center space-x-3">
                          <div className="flex-1 bg-[var(--bg-primary)] h-2.5 rounded-full overflow-hidden border border-[var(--card-border)]">
                            <div 
                              className={`h-full transition-all duration-500 rounded-full ${job.status === 'failed' ? 'bg-rose-500' : 'bg-stone-800 dark:bg-stone-200'}`}
                              style={{ width: `${job.progress}%` }}
                            ></div>
                          </div>
                          <span className="text-xs font-mono text-[var(--text-muted)] w-8 text-right font-medium">{job.progress}%</span>
                        </div>
                      </td>
                      <td className="py-4.5 text-xs text-[var(--text-muted)] pr-3 max-w-xs truncate">
                        {job.error_message || job.progress_message || 'Running...'}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      </main>
    </div>
  );
};
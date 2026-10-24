const API_BASE = 'http://localhost:5000/api';

export const fetchJobs = async () => {
  const response = await fetch(`${API_BASE}/jobs`);
  if (!response.ok) throw new Error('Failed to fetch jobs');
  return response.json();
};

export const startReplayJob = async (videoPath, outputDir) => {
  const response = await fetch(`${API_BASE}/jobs/replay`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ video_path: videoPath, output_dir: outputDir }),
  });
  if (!response.ok) {
    const errData = await response.json().catch(() => ({}));
    throw new Error(errData.error || 'Failed to start replay job');
  }
  return response.json();
};

export const startLiveJob = async (streamUrl, outputDir) => {
  const response = await fetch(`${API_BASE}/jobs/live`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ stream_url: streamUrl, output_dir: outputDir }),
  });
  if (!response.ok) {
    const errData = await response.json().catch(() => ({}));
    throw new Error(errData.error || 'Failed to start live stream job');
  }
  return response.json();
};
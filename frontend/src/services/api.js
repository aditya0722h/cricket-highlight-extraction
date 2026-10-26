const API_BASE = '/api'

async function request(path, options) {
  const response = await fetch(`${API_BASE}${path}`, options)
  const contentType = response.headers.get('content-type') ?? ''
  const payload = contentType.includes('application/json')
    ? await response.json()
    : await response.text()

  if (!response.ok) {
    const message =
      payload && typeof payload === 'object'
        ? payload.error ?? payload.message
        : payload
    throw new Error(message || `Request failed with status ${response.status}`)
  }

  return payload
}

function postJson(path, body) {
  return request(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

export const fetchJobs = () => request('/jobs')

export const startReplayJob = (
  videoPath,
  outputDir = './output_highlights',
  aspect = 'landscape',
  editStyle = 'plain',
  sport = 'cricket',
) =>
  postJson('/jobs/replay', {
    video_path: videoPath,
    output_dir: outputDir,
    aspect,
    edit_style: editStyle,
    sport,
  })

export const startLiveJob = (
  streamUrl,
  outputDir = './output_highlights',
  aspect = 'landscape',
  editStyle = 'plain',
  sport = 'cricket',
) =>
  postJson('/jobs/live', {
    stream_url: streamUrl,
    output_dir: outputDir,
    aspect,
    edit_style: editStyle,
    sport,
  })

export const trainModel = (matchName, groundTruthPath) =>
  postJson('/model/train', {
    match_name: matchName,
    ground_truth_path: groundTruthPath,
  })

export const fetchTrainingLogs = () => request('/training/logs')
import React, { useEffect, useRef, useState } from 'react'

const DEFAULT_API = 'http://127.0.0.1:8000'

async function parseJson(response) {
  const text = await response.text()
  try {
    return text ? JSON.parse(text) : {}
  } catch {
    return { raw: text }
  }
}

function formatMetricLabel(key) {
  return key.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())
}

function formatMetricValue(value) {
  if (typeof value === 'number') {
    if (Number.isInteger(value)) return String(value)
    return value.toFixed(4)
  }
  if (typeof value === 'boolean') return value ? 'true' : 'false'
  if (value === null || value === undefined) return 'n/a'
  return String(value)
}

export default function App() {
  const videoRef = useRef(null)
  const streamRef = useRef(null)
  const canvasRef = useRef(null)

  const [apiBase, setApiBase] = useState(DEFAULT_API)
  const [cameraActive, setCameraActive] = useState(false)

  const [step, setStep] = useState('register')
  const [registeredUser, setRegisteredUser] = useState('')
  const [adminMode, setAdminMode] = useState(false)

  const [enrollUserId, setEnrollUserId] = useState('')
  const [enrollConsent, setEnrollConsent] = useState(true)
  const [enrollFile, setEnrollFile] = useState(null)
  const [enrollCaptureName, setEnrollCaptureName] = useState('')
  const [enrollResult, setEnrollResult] = useState(null)

  const [verifyThreshold, setVerifyThreshold] = useState('0.37')
  const [verifyFile, setVerifyFile] = useState(null)
  const [verifyCaptureName, setVerifyCaptureName] = useState('')
  const [verifyResult, setVerifyResult] = useState(null)

  const [manageUserId, setManageUserId] = useState('')
  const [manageResult, setManageResult] = useState(null)

  const [metricsResult, setMetricsResult] = useState(null)
  const [liveMode, setLiveMode] = useState(false)
  const [liveResult, setLiveResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const liveIntervalRef = useRef(null)

  useEffect(() => {
    return () => {
      stopLiveMode()
      stopCamera()
    }
  }, [])

  async function startCamera() {
    setError('')
    try {
      if (cameraActive && streamRef.current) {
        return
      }
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false })
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play().catch(() => {})
      }
      setCameraActive(true)
    } catch (err) {
      setError(`Camera access failed: ${err.message}`)
    }
  }

  async function waitForVideoReady(timeoutMs = 5000) {
    const start = Date.now()
    while (Date.now() - start < timeoutMs) {
      const video = videoRef.current
      if (video && video.readyState >= 2 && video.videoWidth > 0 && video.videoHeight > 0) {
        return true
      }
      await new Promise((resolve) => setTimeout(resolve, 120))
    }
    return false
  }

  function stopCamera() {
    stopLiveMode()
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop())
      streamRef.current = null
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
    setCameraActive(false)
  }

  async function captureBlobFromVideo() {
    const ready = await waitForVideoReady(4000)
    if (!ready) {
      throw new Error('Camera frame not available yet')
    }

    return new Promise((resolve, reject) => {
      if (!videoRef.current || !canvasRef.current) {
        reject(new Error('Camera not ready'))
        return
      }

      const video = videoRef.current
      const canvas = canvasRef.current

      if (!video.videoWidth || !video.videoHeight) {
        reject(new Error('Camera frame not available yet'))
        return
      }

      canvas.width = video.videoWidth
      canvas.height = video.videoHeight
      const context = canvas.getContext('2d')
      context.drawImage(video, 0, 0, canvas.width, canvas.height)

      canvas.toBlob((blob) => {
        if (!blob) {
          reject(new Error('Failed to capture frame'))
          return
        }
        resolve(blob)
      }, 'image/jpeg', 0.95)
    })
  }

  function captureFrame(target) {
    if (!videoRef.current || !canvasRef.current) {
      setError('Camera not ready')
      return
    }

    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video.videoWidth || !video.videoHeight) {
      setError('Camera is starting. Please wait 1-2 seconds and capture again.')
      return
    }

    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    const context = canvas.getContext('2d')
    context.drawImage(video, 0, 0, canvas.width, canvas.height)

    canvas.toBlob((blob) => {
      if (!blob) {
        setError('Failed to capture frame from camera')
        return
      }
      const timestamp = Date.now()
      const file = new File([blob], `${target}_capture_${timestamp}.jpg`, { type: 'image/jpeg' })

      if (target === 'enroll') {
        setEnrollFile(file)
        setEnrollCaptureName(file.name)
      } else {
        setVerifyFile(file)
        setVerifyCaptureName(file.name)
      }
      setError('')
    }, 'image/jpeg', 0.95)
  }

  async function verifyFromCameraFrame() {
    const blob = await captureBlobFromVideo()
    const file = new File([blob], `live_verify_${Date.now()}.jpg`, { type: 'image/jpeg' })

    const body = new FormData()
    body.append('threshold', verifyThreshold)
    body.append('image', file)

    const response = await fetch(`${apiBase}/verify`, { method: 'POST', body })
    const data = await parseJson(response)
    if (!response.ok) throw new Error(data.detail || 'Live verification failed')

    setLiveResult(data)
    return data
  }

  async function startLiveMode() {
    if (!cameraActive) {
      await startCamera()
    }

    const ready = await waitForVideoReady(5000)
    if (!ready) {
      setError('Camera started, but frame is not ready. Check permission and try again.')
      return
    }

    setError('')
    setLiveMode(true)

    if (liveIntervalRef.current) {
      clearInterval(liveIntervalRef.current)
    }

    await verifyFromCameraFrame().catch((err) => {
      if (!String(err.message || '').toLowerCase().includes('frame not available')) {
        setError(err.message)
      }
    })
    liveIntervalRef.current = setInterval(() => {
      verifyFromCameraFrame().catch((err) => {
        if (!String(err.message || '').toLowerCase().includes('frame not available')) {
          setError(err.message)
        }
      })
    }, 1400)
  }

  async function toggleLiveCamera() {
    if (liveMode || cameraActive) {
      stopLiveMode()
      stopCamera()
      return
    }
    await startLiveMode()
  }

  function stopLiveMode() {
    if (liveIntervalRef.current) {
      clearInterval(liveIntervalRef.current)
      liveIntervalRef.current = null
    }
    setLiveMode(false)
  }

  async function enroll() {
    if (!enrollUserId || !enrollFile) {
      setError('Enrollment requires user_id and image')
      return
    }

    setError('')
    setLoading(true)
    try {
      const body = new FormData()
      body.append('user_id', enrollUserId)
      body.append('consent', String(enrollConsent))
      body.append('image', enrollFile)

      const response = await fetch(`${apiBase}/enroll`, { method: 'POST', body })
      const data = await parseJson(response)
      if (!response.ok) throw new Error(data.detail || 'Enrollment failed')
      setEnrollResult(data)
      setRegisteredUser(data.user_id || enrollUserId)
      stopLiveMode()
      stopCamera()
      setLiveResult(null)
      setStep('unlock')
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  async function verify() {
    if (!verifyFile) {
      setError('Verification requires image')
      return
    }

    setError('')
    setLoading(true)
    try {
      const body = new FormData()
      body.append('threshold', verifyThreshold)
      body.append('image', verifyFile)

      const response = await fetch(`${apiBase}/verify`, { method: 'POST', body })
      const data = await parseJson(response)
      if (!response.ok) throw new Error(data.detail || 'Verification failed')
      setVerifyResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  async function getUser() {
    if (!manageUserId) {
      setError('Enter user id')
      return
    }

    setError('')
    setLoading(true)
    try {
      const response = await fetch(`${apiBase}/users/${manageUserId}`)
      const data = await parseJson(response)
      if (!response.ok) throw new Error(data.detail || 'User fetch failed')
      setManageResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  async function revokeConsent() {
    if (!manageUserId) {
      setError('Enter user id')
      return
    }

    setError('')
    setLoading(true)
    try {
      const response = await fetch(`${apiBase}/users/${manageUserId}/revoke`, {
        method: 'POST'
      })
      const data = await parseJson(response)
      if (!response.ok) throw new Error(data.detail || 'Revoke failed')
      setManageResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  async function deleteUser() {
    if (!manageUserId) {
      setError('Enter user id')
      return
    }

    setError('')
    setLoading(true)
    try {
      const response = await fetch(`${apiBase}/users/${manageUserId}`, {
        method: 'DELETE'
      })
      const data = await parseJson(response)
      if (!response.ok) throw new Error(data.detail || 'Delete failed')
      setManageResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  async function loadMetrics() {
    setError('')
    setLoading(true)
    try {
      const response = await fetch(`${apiBase}/metrics`)
      const data = await parseJson(response)
      if (!response.ok) throw new Error(data.detail || 'Metrics fetch failed')
      setMetricsResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="container">
      <header className="hero">
        <h1>Privacy-Preserving Face Recognition</h1>
        <p className="subtitle">Register with camera + consent, then test live face unlock in real time.</p>
      </header>

      <section className="card card-full">
        <h2>API Settings</h2>
        <label>API Base URL</label>
        <input value={apiBase} onChange={(e) => setApiBase(e.target.value)} placeholder="http://127.0.0.1:8000" />
        <div className="row">
          <button className="btn" onClick={() => setAdminMode((prev) => !prev)}>
            {adminMode ? 'Disable Admin Mode' : 'Enable Admin Mode'}
          </button>
          <span className="chip">Admin Mode: {adminMode ? 'ON' : 'OFF'}</span>
        </div>
      </section>

      {step === 'register' && (
        <section className="card card-full">
          <h2>Step 1: Register Your Face</h2>
          <p className="hint">Open the camera, capture your face, provide name and consent, then register.</p>
          <div className="register-layout">
            <div>
              <div className="camera-wrap">
                <video ref={videoRef} autoPlay playsInline muted className="camera" />
                <canvas ref={canvasRef} style={{ display: 'none' }} />
              </div>
              <div className="row camera-controls">
                {!cameraActive ? (
                  <button className="btn btn-primary" onClick={startCamera} disabled={loading}>Open Camera</button>
                ) : (
                  <button className="btn" onClick={stopCamera} disabled={loading}>Stop Camera</button>
                )}
                <button className="btn" onClick={() => captureFrame('enroll')} disabled={!cameraActive || loading}>Capture Face</button>
              </div>
              <span className="chip capture-pill">Capture: {enrollCaptureName || 'not captured'}</span>
            </div>

            <div className="form-stack">
              <label>Name / User ID</label>
              <input value={enrollUserId} onChange={(e) => setEnrollUserId(e.target.value)} placeholder="e.g. bruce_01" />

              <label>Consent</label>
              <select value={String(enrollConsent)} onChange={(e) => setEnrollConsent(e.target.value === 'true')}>
                <option value="true">I consent (required)</option>
                <option value="false">I do not consent</option>
              </select>

              <label>Or upload face image</label>
              <input type="file" accept="image/*" onChange={(e) => {
                const file = e.target.files?.[0] || null
                setEnrollFile(file)
                setEnrollCaptureName(file ? file.name : '')
              }} />

              <button className="btn btn-primary" onClick={enroll} disabled={loading}>Register Face</button>
              {enrollResult && <pre>{JSON.stringify(enrollResult, null, 2)}</pre>}
            </div>
          </div>
        </section>
      )}

      {step === 'unlock' && (
        <section className="card card-full">
          <div className="section-head">
            <h2>Step 2: Live Face Unlock</h2>
            <div className="status-row">
              <span className={`status-pill ${cameraActive ? 'on' : 'off'}`}>
                <span className="status-dot" />
                Camera {cameraActive ? 'On' : 'Off'}
              </span>
              <span className={`status-pill ${liveMode ? 'on' : 'off'}`}>
                <span className="status-dot" />
                Live Detection {liveMode ? 'On' : 'Off'}
              </span>
            </div>
          </div>
          <p className="hint">User <strong>{registeredUser || 'registered user'}</strong>. Live score updates every ~1.4s.</p>
          <div className="unlock-layout">
            <div>
              <div className="camera-wrap">
                <video ref={videoRef} autoPlay playsInline muted className="camera" />
                <canvas ref={canvasRef} style={{ display: 'none' }} />
              </div>
              <div className="row camera-controls">
                <button className="btn btn-primary" onClick={toggleLiveCamera} disabled={loading}>
                  {liveMode || cameraActive ? 'Stop Live Camera' : 'Start Live Camera'}
                </button>
                <button className="btn" onClick={() => setStep('register')}>Back to Registration</button>
              </div>
            </div>

            <div className="result-box">
              <label>Threshold</label>
              <input value={verifyThreshold} onChange={(e) => setVerifyThreshold(e.target.value)} placeholder="0.37" />
              <label>Upload probe (optional)</label>
              <input type="file" accept="image/*" onChange={(e) => {
                const file = e.target.files?.[0] || null
                setVerifyFile(file)
                setVerifyCaptureName(file ? file.name : '')
              }} />
              <button className="btn" onClick={verify} disabled={loading || !verifyFile}>Verify Uploaded Probe</button>
              <span className="chip">Upload: {verifyCaptureName || 'none'}</span>

              <div className={`live-status ${(liveResult && liveResult.verified) ? 'ok' : 'warn'}`}>
                <div className="live-title">Live Status</div>
                <div>Verified: {String(liveResult?.verified ?? false)}</div>
                <div>Score: {typeof liveResult?.confidence === 'number' ? liveResult.confidence.toFixed(3) : 'n/a'}</div>
                <div>Matched User: {liveResult?.matched_user_id || 'none'}</div>
              </div>

              {verifyResult && (
                <div className="live-status">
                  <div className="live-title">Upload Verify Result</div>
                  <div>Verified: {String(verifyResult?.verified ?? false)}</div>
                  <div>Score: {typeof verifyResult?.confidence === 'number' ? verifyResult.confidence.toFixed(3) : 'n/a'}</div>
                  <div>Matched User: {verifyResult?.matched_user_id || 'none'}</div>
                </div>
              )}
            </div>
          </div>
        </section>
      )}

      {adminMode && (
        <div className="grid">
          <section className="card">
            <h2>Consent Management</h2>
            <label>User ID</label>
            <input value={manageUserId} onChange={(e) => setManageUserId(e.target.value)} placeholder="e.g. user_1" />
            <div className="row">
              <button className="btn" onClick={getUser} disabled={loading}>Get User</button>
              <button className="btn" onClick={revokeConsent} disabled={loading}>Revoke Consent</button>
              <button className="btn btn-danger" onClick={deleteUser} disabled={loading}>Delete User</button>
            </div>
            {manageResult && <pre>{JSON.stringify(manageResult, null, 2)}</pre>}
          </section>

          <section className="card">
            <h2>Metrics Dashboard</h2>
            <button className="btn" onClick={loadMetrics} disabled={loading}>Load Metrics</button>
            {metricsResult && (
              <div className="metrics-stack">
                <div>
                  <h3 className="mini-title">Store Metrics</h3>
                  <div className="metric-grid">
                    {Object.entries(metricsResult.store_metrics || {}).map(([key, value]) => (
                      <div key={key} className="metric-item">
                        <div className="metric-label">{formatMetricLabel(key)}</div>
                        <div className="metric-value">{formatMetricValue(value)}</div>
                      </div>
                    ))}
                  </div>
                </div>

                <div>
                  <h3 className="mini-title">Evaluation Metrics</h3>
                  {metricsResult.evaluation_metrics ? (
                    <div className="metric-grid">
                      {Object.entries(metricsResult.evaluation_metrics).map(([key, value]) => (
                        <div key={key} className="metric-item">
                          <div className="metric-label">{formatMetricLabel(key)}</div>
                          <div className="metric-value">{formatMetricValue(value)}</div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="hint">No evaluation metrics found yet.</p>
                  )}
                </div>
              </div>
            )}
          </section>
        </div>
      )}

      {error && <p className="error">{error}</p>}
    </main>
  )
}

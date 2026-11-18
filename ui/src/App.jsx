
import { useEffect, useState } from "react";
import {
  Activity,
  ArrowRight,
  CheckCircle2,
  Clock3,
  Gauge,
  MapPin,
  Navigation,
  Play,
  Route,
  Server,
  ShieldCheck,
  Zap,
} from "lucide-react";
import {
  CircleMarker,
  MapContainer,
  Polyline,
  TileLayer,
  useMap,
} from "react-leaflet";
import "leaflet/dist/leaflet.css";
import "./App.css";

const API = "http://localhost:8000";

function FitMap({ gps, route }) {
  const map = useMap();

  useEffect(() => {
    const points = [
      ...gps,
      ...route.flat(),
    ].filter(
      (p) =>
        Array.isArray(p) &&
        p.length >= 2 &&
        Number.isFinite(p[0]) &&
        Number.isFinite(p[1])
    );

    if (points.length) {
      map.fitBounds(points, { padding: [35, 35], maxZoom: 16 });
    }
  }, [gps, route, map]);

  return null;
}

function formatMetric(value) {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") {
    return value <= 1 ? `${(value * 100).toFixed(1)}%` : value.toFixed(3);
  }
  return String(value);
}

function App() {
  const [trips, setTrips] = useState([]);
  const [selectedTrip, setSelectedTrip] = useState("");
  const [useHmm, setUseHmm] = useState(true);
  const [health, setHealth] = useState(false);
  const [loadingTrips, setLoadingTrips] = useState(true);
  const [predicting, setPredicting] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    async function initialise() {
      try {
        const [healthResponse, tripsResponse] = await Promise.all([
          fetch(`${API}/health`),
          fetch(`${API}/trips`),
        ]);

        if (!healthResponse.ok || !tripsResponse.ok) {
          throw new Error("Could not connect to the PathSnap API.");
        }

        const healthData = await healthResponse.json();
        const tripsData = await tripsResponse.json();

        setHealth(healthData.status === "ok");
        setTrips(tripsData.trips || []);

        if (tripsData.trips?.length) {
          setSelectedTrip(tripsData.trips[0]);
        }
      } catch (err) {
        setError(`${err.message} Make sure the FastAPI server is running.`);
      } finally {
        setLoadingTrips(false);
      }
    }

    initialise();
  }, []);

  async function runPrediction() {
    if (!selectedTrip) return;

    setPredicting(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(`${API}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          trip_id: selectedTrip,
          use_hmm: useHmm,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Prediction failed.");
      }

      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setPredicting(false);
    }
  }

  const gps = result?.gps_points || [];
  const predictedRoute = result?.predicted_route || [];
  const metrics = result?.metrics || {};
  const timings = result?.timings_s || {};

  const mapCenter = gps.length ? gps[0] : [12.9716, 77.5946];

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-icon">
            <Navigation size={21} />
          </div>
          <div>
            <h1>PathSnap</h1>
            <span>Intelligent GPS Map Matching</span>
          </div>
        </div>

        <div className="topbar-right">
          <div className={`connection ${health ? "online" : "offline"}`}>
            <span className="status-dot" />
            {health ? "API Connected" : "API Offline"}
          </div>
          <div className="version">v1.0</div>
        </div>
      </header>

      <main className="dashboard">
        <section className="welcome">
          <div>
            <p className="eyebrow">MAP MATCHING WORKSPACE</p>
            <h2>Turn GPS traces into road routes.</h2>
            <p className="welcome-text">
              Select a trip, configure the decoder, and reconstruct its most
              likely path through the road network.
            </p>
          </div>
          <div className="welcome-icon">
            <Route size={42} strokeWidth={1.4} />
          </div>
        </section>

        <section className="workspace">
          <aside className="control-panel panel">
            <div className="panel-heading">
              <div className="heading-icon"><Activity size={18} /></div>
              <div>
                <h3>Prediction Controls</h3>
                <p>Configure your map matching run</p>
              </div>
            </div>

            <label className="field-label" htmlFor="trip">
              SELECT TRIP
            </label>
            <select
              id="trip"
              value={selectedTrip}
              onChange={(e) => {
                setSelectedTrip(e.target.value);
                setResult(null);
              }}
              disabled={loadingTrips || !trips.length}
            >
              {loadingTrips && <option>Loading trips...</option>}
              {!loadingTrips && !trips.length && (
                <option value="">No trips available</option>
              )}
              {trips.map((trip) => (
                <option key={trip} value={trip}>{trip}</option>
              ))}
            </select>

            <div className="decoder-option">
              <div className="decoder-copy">
                <div className="decoder-title">
                  <ShieldCheck size={17} />
                  <span>HMM / Viterbi Decoder</span>
                </div>
                <p>Use sequential route optimisation</p>
              </div>
              <label className="switch">
                <input
                  type="checkbox"
                  checked={useHmm}
                  onChange={(e) => setUseHmm(e.target.checked)}
                />
                <span className="switch-slider" />
              </label>
            </div>

            <button
              className="predict-button"
              onClick={runPrediction}
              disabled={!selectedTrip || predicting || !health}
            >
              {predicting ? (
                <><span className="spinner" /> Processing...</>
              ) : (
                <><Play size={17} fill="currentColor" /> Run Map Matching</>
              )}
            </button>

            <div className="control-note">
              <Zap size={15} />
              <span>
                {useHmm
                  ? "Classifier + HMM decoding enabled"
                  : "Classifier-only prediction enabled"}
              </span>
            </div>

            {error && <div className="error-message">{error}</div>}

            <div className="panel-divider" />

            <div className="info-row">
              <Server size={16} />
              <span>Backend service</span>
              <strong>{health ? "Online" : "Offline"}</strong>
            </div>
            <div className="info-row">
              <MapPin size={16} />
              <span>Available trips</span>
              <strong>{loadingTrips ? "..." : trips.length}</strong>
            </div>
          </aside>

          <section className="map-panel panel">
            <div className="map-heading">
              <div>
                <h3>Route Visualization</h3>
                <p>
                  {result
                    ? `Showing prediction for ${result.trip_id}`
                    : "Your reconstructed route will appear here"}
                </p>
              </div>
              <div className="map-live">
                <span className="status-dot" />
                LIVE MAP
              </div>
            </div>

            <div className="map-container">
              <MapContainer
                center={mapCenter}
                zoom={12}
                scrollWheelZoom
                className="leaflet-map"
              >
                <TileLayer
                  attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                  url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                />

                {gps.length > 0 && (
                  <>
                    <Polyline
                      positions={gps}
                      pathOptions={{
                        color: "#e7a83e",
                        weight: 3,
                        opacity: 0.8,
                        dashArray: "6 7",
                      }}
                    />
                    {gps.map((point, index) => (
                      <CircleMarker
                        key={index}
                        center={point}
                        radius={index === 0 || index === gps.length - 1 ? 5 : 2.5}
                        pathOptions={{
                          color: "#ffffff",
                          weight: 1,
                          fillColor: "#e7a83e",
                          fillOpacity: 0.9,
                        }}
                      />
                    ))}
                  </>
                )}

                {predictedRoute.map((segment, index) => (
                  <Polyline
                    key={index}
                    positions={segment}
                    pathOptions={{
                      color: "#2868e8",
                      weight: 5,
                      opacity: 0.95,
                    }}
                  />
                ))}

                {gps.length > 0 && (
                  <>
                    <CircleMarker
                      center={gps[0]}
                      radius={7}
                      pathOptions={{
                        color: "#ffffff",
                        weight: 2,
                        fillColor: "#16a36a",
                        fillOpacity: 1,
                      }}
                    />
                    <CircleMarker
                      center={gps[gps.length - 1]}
                      radius={7}
                      pathOptions={{
                        color: "#ffffff",
                        weight: 2,
                        fillColor: "#dc5353",
                        fillOpacity: 1,
                      }}
                    />
                  </>
                )}

                <FitMap gps={gps} route={predictedRoute} />
              </MapContainer>

              {!result && (
                <div className="map-placeholder">
                  <div className="placeholder-icon">
                    <MapPin size={27} />
                  </div>
                  <h3>Ready to map your route</h3>
                  <p>Select a trip and run map matching to visualize the result.</p>
                </div>
              )}
            </div>

            <div className="map-legend">
              <span><i className="legend-gps" /> GPS trajectory</span>
              <span><i className="legend-route" /> Predicted route</span>
              <span><i className="legend-start" /> Start</span>
              <span><i className="legend-end" /> End</span>
            </div>
          </section>
        </section>

        <section className="results-section">
          <div className="section-title">
            <div>
              <p className="eyebrow">ANALYSIS</p>
              <h3>Prediction Summary</h3>
            </div>
            {result && (
              <div className="success-label">
                <CheckCircle2 size={15} /> Prediction complete
              </div>
            )}
          </div>

          <div className="metrics-grid">
            <Metric icon={<Gauge />} label="Mean Confidence" value={
              result
                ? `${(result.mean_point_confidence * 100).toFixed(1)}%`
                : "—"
            } />
            <Metric icon={<MapPin />} label="GPS Observations" value={
              result ? result.n_points.toLocaleString() : "—"
            } />
            <Metric icon={<Route />} label="Route Segments" value={
              result ? result.predicted_route_arc_ids?.length ?? 0 : "—"
            } />
            <Metric icon={<Clock3 />} label="Prediction Time" value={
              timings.predict_total !== undefined
                ? `${timings.predict_total.toFixed(3)}s`
                : "—"
            } />
          </div>

          {result && (
            <div className="details-panel panel">
              <div className="details-heading">
                <h3>Evaluation & Processing Details</h3>
                <span>{result.decoder}</span>
              </div>

              <div className="details-grid">
                {Object.entries(metrics).map(([key, value]) => (
                  <div className="detail-item" key={key}>
                    <span>{key.replaceAll("_", " ")}</span>
                    <strong>{formatMetric(value)}</strong>
                  </div>
                ))}
                <div className="detail-item">
                  <span>HMM breaks</span>
                  <strong>{result.hmm_breaks ?? 0}</strong>
                </div>
                <div className="detail-item">
                  <span>Unconnected gaps</span>
                  <strong>{result.unconnected_gaps ?? 0}</strong>
                </div>
              </div>

              <div className="timing-row">
                {Object.entries(timings).map(([key, value]) => (
                  <div key={key}>
                    <span>{key.replaceAll("_", " ")}</span>
                    <strong>{Number(value).toFixed(4)} s</strong>
                  </div>
                ))}
              </div>

              <details className="arc-details">
                <summary>View predicted arc IDs</summary>
                <p>{(result.predicted_route_arc_ids || []).join(", ") || "No arc IDs returned."}</p>
              </details>
            </div>
          )}

          {!result && (
            <div className="empty-results panel">
              <Activity size={21} />
              <span>Run a prediction to see route metrics and processing details.</span>
              <ArrowRight size={17} />
            </div>
          )}
        </section>

        <footer>
          <span>PathSnap · Intelligent GPS Map Matching</span>
          <span>Powered by Machine Learning & Hidden Markov Models</span>
        </footer>
      </main>
    </div>
  );
}

function Metric({ icon, label, value }) {
  return (
    <div className="metric-card panel">
      <div className="metric-icon">{icon}</div>
      <p>{label}</p>
      <strong>{value}</strong>
    </div>
  );
}

export default App;

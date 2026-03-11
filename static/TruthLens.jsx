import { useState, useEffect, useRef } from "react";

const API_URL = "http://localhost:8000";

// ─── Color Tokens ──────────────────────────────────────────────────────────
const T = {
  bg: "#0a0e17",
  surface: "#111827",
  surfaceHover: "#1a2332",
  border: "#1e293b",
  borderFocus: "#3b82f6",
  text: "#e2e8f0",
  textMuted: "#64748b",
  textDim: "#475569",
  accent: "#3b82f6",
  accentGlow: "rgba(59,130,246,0.15)",
  green: "#22c55e",
  greenDim: "rgba(34,197,94,0.12)",
  red: "#ef4444",
  redDim: "rgba(239,68,68,0.12)",
  yellow: "#eab308",
  yellowDim: "rgba(234,179,8,0.12)",
  purple: "#a855f7",
  purpleDim: "rgba(168,85,247,0.12)",
};

// ─── Reliability Score Ring ────────────────────────────────────────────────
function ScoreRing({ score, size = 140 }) {
  const radius = (size - 16) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - score * circumference;
  const color = score >= 0.7 ? T.green : score >= 0.4 ? T.yellow : T.red;
  const [animated, setAnimated] = useState(false);

  useEffect(() => {
    const t = setTimeout(() => setAnimated(true), 100);
    return () => clearTimeout(t);
  }, [score]);

  return (
    <div style={{ position: "relative", width: size, height: size }}>
      <svg width={size} height={size} style={{ transform: "rotate(-90deg)" }}>
        <circle
          cx={size / 2} cy={size / 2} r={radius}
          fill="none" stroke={T.border} strokeWidth="6"
        />
        <circle
          cx={size / 2} cy={size / 2} r={radius}
          fill="none" stroke={color} strokeWidth="6"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={animated ? offset : circumference}
          style={{ transition: "stroke-dashoffset 1.2s cubic-bezier(0.4, 0, 0.2, 1)" }}
        />
      </svg>
      <div style={{
        position: "absolute", inset: 0,
        display: "flex", flexDirection: "column",
        alignItems: "center", justifyContent: "center",
      }}>
        <span style={{ fontSize: 32, fontWeight: 700, color, fontFamily: "'JetBrains Mono', monospace" }}>
          {Math.round(score * 100)}%
        </span>
        <span style={{ fontSize: 10, color: T.textMuted, letterSpacing: 1.5, textTransform: "uppercase", marginTop: 2 }}>
          reliability
        </span>
      </div>
    </div>
  );
}

// ─── Claim Card ────────────────────────────────────────────────────────────
function ClaimCard({ verification, index }) {
  const [expanded, setExpanded] = useState(false);
  const s = verification.status;
  const conf = verification.confidence;

  const statusConfig = {
    supported: { icon: "✓", label: "Supported", color: T.green, bg: T.greenDim },
    contradicted: { icon: "✗", label: "Contradicted", color: T.red, bg: T.redDim },
    unverifiable: { icon: "?", label: "Unverifiable", color: T.yellow, bg: T.yellowDim },
    partially_supported: { icon: "~", label: "Partial", color: T.purple, bg: T.purpleDim },
  };
  const cfg = statusConfig[s] || statusConfig.unverifiable;

  return (
    <div
      onClick={() => setExpanded(!expanded)}
      style={{
        background: T.surface,
        border: `1px solid ${T.border}`,
        borderLeft: `3px solid ${cfg.color}`,
        borderRadius: 8,
        padding: "14px 18px",
        cursor: "pointer",
        transition: "all 0.2s",
        animationName: "slideIn",
        animationDuration: "0.4s",
        animationTimingFunction: "cubic-bezier(0.16, 1, 0.3, 1)",
        animationFillMode: "both",
        animationDelay: `${index * 80}ms`,
      }}
      onMouseEnter={e => { e.currentTarget.style.background = T.surfaceHover; e.currentTarget.style.borderColor = cfg.color; }}
      onMouseLeave={e => { e.currentTarget.style.background = T.surface; e.currentTarget.style.borderColor = T.border; }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        <div style={{
          width: 28, height: 28, borderRadius: 6,
          background: cfg.bg, color: cfg.color,
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: 14, fontWeight: 700, fontFamily: "'JetBrains Mono', monospace",
          flexShrink: 0,
        }}>
          {cfg.icon}
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontSize: 13, color: T.text, lineHeight: 1.5 }}>
            {verification.claim.text}
          </div>
        </div>
        <div style={{
          display: "flex", alignItems: "center", gap: 8, flexShrink: 0,
        }}>
          <span style={{
            fontSize: 11, color: cfg.color, background: cfg.bg,
            padding: "3px 8px", borderRadius: 4, fontWeight: 600,
            fontFamily: "'JetBrains Mono', monospace",
          }}>
            {cfg.label}
          </span>
          <span style={{
            fontSize: 11, color: T.textMuted,
            fontFamily: "'JetBrains Mono', monospace",
          }}>
            {Math.round(conf * 100)}%
          </span>
          <span style={{
            fontSize: 10, color: T.textDim,
            transform: expanded ? "rotate(180deg)" : "rotate(0)",
            transition: "transform 0.2s",
          }}>▼</span>
        </div>
      </div>

      {expanded && verification.reasoning && (
        <div style={{
          marginTop: 12, paddingTop: 12,
          borderTop: `1px solid ${T.border}`,
          fontSize: 12, color: T.textMuted, lineHeight: 1.6,
        }}>
          <span style={{ color: T.textDim, fontWeight: 600, fontSize: 10, textTransform: "uppercase", letterSpacing: 1 }}>
            Reasoning
          </span>
          <p style={{ margin: "6px 0 0" }}>{verification.reasoning}</p>

          {verification.evidence && verification.evidence.length > 0 && (
            <div style={{ marginTop: 10 }}>
              <span style={{ color: T.textDim, fontWeight: 600, fontSize: 10, textTransform: "uppercase", letterSpacing: 1 }}>
                Evidence ({verification.evidence.length} sources)
              </span>
              {verification.evidence.slice(0, 2).map((e, i) => (
                <div key={i} style={{
                  marginTop: 6, padding: "8px 10px",
                  background: T.bg, borderRadius: 4,
                  fontSize: 11, color: T.textMuted, lineHeight: 1.5,
                  borderLeft: `2px solid ${T.textDim}`,
                }}>
                  <span style={{ color: T.textDim, fontSize: 10 }}>[{e.chunk.source}]</span>{" "}
                  {e.chunk.text.slice(0, 200)}{e.chunk.text.length > 200 ? "…" : ""}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ─── Stats Bar ─────────────────────────────────────────────────────────────
function StatsBar({ response }) {
  const stats = [
    { label: "Claims", value: response.total_claims, color: T.accent },
    { label: "Supported", value: response.supported_claims, color: T.green },
    { label: "Contradicted", value: response.contradicted_claims, color: T.red },
    { label: "Unverifiable", value: response.unverifiable_claims, color: T.yellow },
    { label: "Time", value: `${(response.verification_time_ms / 1000).toFixed(1)}s`, color: T.textMuted },
  ];

  return (
    <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
      {stats.map((s, i) => (
        <div key={i} style={{
          display: "flex", alignItems: "center", gap: 6,
          padding: "6px 12px", background: T.surface,
          border: `1px solid ${T.border}`, borderRadius: 6,
        }}>
          <span style={{ fontSize: 11, color: T.textMuted }}>{s.label}</span>
          <span style={{ fontSize: 13, fontWeight: 700, color: s.color, fontFamily: "'JetBrains Mono', monospace" }}>
            {s.value}
          </span>
        </div>
      ))}
    </div>
  );
}

// ─── Loading Dots ──────────────────────────────────────────────────────────
function LoadingDots() {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 12, padding: "40px 0" }}>
      <div style={{ display: "flex", gap: 4 }}>
        {[0, 1, 2].map(i => (
          <div key={i} style={{
            width: 6, height: 6, borderRadius: "50%",
            background: T.accent,
            animation: "pulse 1.2s ease-in-out infinite",
            animationDelay: `${i * 0.15}s`,
          }} />
        ))}
      </div>
      <span style={{ fontSize: 13, color: T.textMuted }}>
        Extracting claims, retrieving evidence, cross-checking…
      </span>
    </div>
  );
}

// ─── Main App ──────────────────────────────────────────────────────────────
export default function TruthLens() {
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [serverStatus, setServerStatus] = useState("checking");
  const [stats, setStats] = useState(null);
  const textareaRef = useRef(null);

  // Check server on mount
  useEffect(() => {
    fetch(`${API_URL}/health`)
      .then(r => r.json())
      .then(d => {
        setServerStatus("connected");
        return fetch(`${API_URL}/stats`);
      })
      .then(r => r.json())
      .then(d => setStats(d))
      .catch(() => setServerStatus("disconnected"));
  }, []);

  const handleVerify = async () => {
    if (!input.trim() || loading) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch(`${API_URL}/verify`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ llm_response: input }),
      });
      if (!res.ok) throw new Error(`Server returned ${res.status}`);
      const data = await res.json();
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      handleVerify();
    }
  };

  const exampleTexts = [
    "The Eiffel Tower is 324 meters tall and was built in 1889 for the World's Fair in Paris, France. It is made of wrought iron.",
    "The Great Wall of China is over 13,000 miles long, was built during the Qin dynasty, and is visible from space with the naked eye.",
    "Python was created by Guido van Rossum and first released in 1995. It is a statically typed language that uses curly braces for code blocks.",
  ];

  return (
    <div style={{
      minHeight: "100vh",
      background: T.bg,
      color: T.text,
      fontFamily: "'IBM Plex Sans', -apple-system, sans-serif",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600;700&display=swap');

        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::selection { background: ${T.accent}; color: white; }
        textarea:focus { outline: none; }

        @keyframes slideIn {
          from { opacity: 0; transform: translateY(8px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes pulse {
          0%, 100% { opacity: 0.3; transform: scale(0.8); }
          50% { opacity: 1; transform: scale(1); }
        }
        @keyframes fadeIn {
          from { opacity: 0; }
          to { opacity: 1; }
        }
        @keyframes gradientShift {
          0% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
          100% { background-position: 0% 50%; }
        }
      `}</style>

      {/* ─── Header ───────────────────────────────────────────────── */}
      <header style={{
        borderBottom: `1px solid ${T.border}`,
        padding: "16px 0",
      }}>
        <div style={{ maxWidth: 900, margin: "0 auto", padding: "0 24px", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <div style={{
              width: 32, height: 32, borderRadius: 8,
              background: `linear-gradient(135deg, ${T.accent}, #8b5cf6)`,
              display: "flex", alignItems: "center", justifyContent: "center",
              fontSize: 16,
            }}>🔍</div>
            <div>
              <h1 style={{
                fontSize: 18, fontWeight: 700, letterSpacing: -0.5,
                background: `linear-gradient(135deg, ${T.text}, ${T.accent})`,
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: "transparent",
              }}>
                TruthLens
              </h1>
              <p style={{ fontSize: 11, color: T.textDim, marginTop: -1 }}>LLM Hallucination Detection</p>
            </div>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            {stats && (
              <span style={{
                fontSize: 11, color: T.textDim,
                fontFamily: "'JetBrains Mono', monospace",
              }}>
                {stats.index?.num_vectors || 0} vectors indexed
              </span>
            )}
            <div style={{
              display: "flex", alignItems: "center", gap: 6,
              padding: "4px 10px", borderRadius: 20,
              background: serverStatus === "connected" ? T.greenDim : serverStatus === "disconnected" ? T.redDim : T.surface,
              border: `1px solid ${serverStatus === "connected" ? T.green : serverStatus === "disconnected" ? T.red : T.border}`,
            }}>
              <div style={{
                width: 6, height: 6, borderRadius: "50%",
                background: serverStatus === "connected" ? T.green : serverStatus === "disconnected" ? T.red : T.yellow,
              }} />
              <span style={{
                fontSize: 11, fontWeight: 600,
                color: serverStatus === "connected" ? T.green : serverStatus === "disconnected" ? T.red : T.yellow,
              }}>
                {serverStatus === "connected" ? "Ollama Connected" : serverStatus === "disconnected" ? "Server Offline" : "Checking…"}
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* ─── Main Content ─────────────────────────────────────────── */}
      <main style={{ maxWidth: 900, margin: "0 auto", padding: "32px 24px" }}>

        {/* Input Area */}
        <div style={{ marginBottom: 32 }}>
          <label style={{
            display: "block", fontSize: 12, fontWeight: 600,
            color: T.textMuted, marginBottom: 8,
            letterSpacing: 0.5, textTransform: "uppercase",
          }}>
            Paste LLM response to verify
          </label>
          <div style={{
            position: "relative",
            border: `1px solid ${T.border}`,
            borderRadius: 10,
            background: T.surface,
            transition: "border-color 0.2s",
          }}
          onFocus={e => e.currentTarget.style.borderColor = T.borderFocus}
          onBlur={e => e.currentTarget.style.borderColor = T.border}
          >
            <textarea
              ref={textareaRef}
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="The Eiffel Tower is 324 meters tall and was built in 1889…"
              rows={5}
              style={{
                width: "100%", padding: "16px 18px",
                background: "transparent", border: "none",
                color: T.text, fontSize: 14, lineHeight: 1.7,
                fontFamily: "'IBM Plex Sans', sans-serif",
                resize: "vertical", minHeight: 120,
              }}
            />
            <div style={{
              display: "flex", alignItems: "center", justifyContent: "space-between",
              padding: "10px 18px", borderTop: `1px solid ${T.border}`,
            }}>
              <span style={{ fontSize: 11, color: T.textDim }}>
                {input.length > 0 ? `${input.length} chars` : "⌘+Enter to verify"}
              </span>
              <button
                onClick={handleVerify}
                disabled={loading || !input.trim() || serverStatus !== "connected"}
                style={{
                  padding: "8px 20px", borderRadius: 6,
                  background: loading || !input.trim() ? T.textDim : T.accent,
                  color: "white", border: "none",
                  fontSize: 13, fontWeight: 600,
                  cursor: loading || !input.trim() ? "not-allowed" : "pointer",
                  transition: "all 0.2s",
                  fontFamily: "'IBM Plex Sans', sans-serif",
                }}
              >
                {loading ? "Verifying…" : "Verify Claims"}
              </button>
            </div>
          </div>

          {/* Example buttons */}
          <div style={{ marginTop: 10, display: "flex", gap: 6, flexWrap: "wrap" }}>
            <span style={{ fontSize: 11, color: T.textDim, alignSelf: "center", marginRight: 4 }}>Try:</span>
            {["Eiffel Tower", "Great Wall", "Python"].map((label, i) => (
              <button key={i} onClick={() => setInput(exampleTexts[i])} style={{
                padding: "4px 10px", borderRadius: 4,
                background: "transparent", border: `1px solid ${T.border}`,
                color: T.textMuted, fontSize: 11, cursor: "pointer",
                transition: "all 0.15s",
                fontFamily: "'IBM Plex Sans', sans-serif",
              }}
              onMouseEnter={e => { e.target.style.borderColor = T.accent; e.target.style.color = T.accent; }}
              onMouseLeave={e => { e.target.style.borderColor = T.border; e.target.style.color = T.textMuted; }}
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        {/* Error */}
        {error && (
          <div style={{
            padding: "12px 16px", borderRadius: 8,
            background: T.redDim, border: `1px solid ${T.red}`,
            color: T.red, fontSize: 13, marginBottom: 24,
          }}>
            {error === "Failed to fetch"
              ? "Cannot reach the TruthLens server. Make sure it's running on localhost:8000."
              : error}
          </div>
        )}

        {/* Loading */}
        {loading && <LoadingDots />}

        {/* Results */}
        {result && !loading && (
          <div style={{ animation: "fadeIn 0.3s ease" }}>
            {/* Score + Stats Row */}
            <div style={{
              display: "flex", alignItems: "center", gap: 28,
              marginBottom: 28, padding: "24px",
              background: T.surface, borderRadius: 12,
              border: `1px solid ${T.border}`,
            }}>
              <ScoreRing score={result.overall_reliability_score} />
              <div style={{ flex: 1 }}>
                <h3 style={{ fontSize: 16, fontWeight: 600, marginBottom: 4 }}>
                  Verification Report
                </h3>
                <p style={{ fontSize: 12, color: T.textMuted, marginBottom: 14 }}>
                  {result.total_claims} claims extracted and cross-checked against {stats?.index?.num_vectors || "the"} knowledge base vectors
                </p>
                <StatsBar response={result} />
              </div>
            </div>

            {/* Claim Cards */}
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              <span style={{
                fontSize: 12, fontWeight: 600, color: T.textMuted,
                letterSpacing: 0.5, textTransform: "uppercase", marginBottom: 4,
              }}>
                Claim-by-claim analysis
              </span>
              {result.claim_verifications.map((v, i) => (
                <ClaimCard key={v.claim.claim_id} verification={v} index={i} />
              ))}
            </div>
          </div>
        )}

        {/* Empty State */}
        {!result && !loading && !error && (
          <div style={{
            textAlign: "center", padding: "60px 0",
            color: T.textDim, fontSize: 13,
          }}>
            <div style={{ fontSize: 40, marginBottom: 12, opacity: 0.4 }}>🔬</div>
            <p style={{ marginBottom: 4 }}>Paste any LLM-generated text above to check for hallucinations</p>
            <p style={{ fontSize: 11, color: T.textDim }}>
              Claims are extracted, evidence is retrieved from FAISS, and each claim is cross-verified
            </p>
          </div>
        )}
      </main>

      {/* ─── Footer ───────────────────────────────────────────────── */}
      <footer style={{
        borderTop: `1px solid ${T.border}`,
        padding: "16px 0", marginTop: 40,
        textAlign: "center",
      }}>
        <p style={{ fontSize: 11, color: T.textDim }}>
          TruthLens — FAISS + Ollama • Built with FastAPI & React
        </p>
      </footer>
    </div>
  );
}

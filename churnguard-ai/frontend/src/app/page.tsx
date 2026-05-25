"use client";

import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { ArrowTrendingUpIcon, BoltIcon, ShieldCheckIcon } from "@heroicons/react/24/solid";
import type { BatchRow, ExplainItem, PredictPayload } from "../lib/api";
import { explainCustomer, predictBatch } from "../lib/api";

const EMPTY_EXPLAIN: ExplainItem[] = [];

const lowRiskSample: PredictPayload = {
  tenure: 56,
  MonthlyCharges: 42,
  TotalCharges: 2352,
  SeniorCitizen: 0,
  Partner: "Yes",
  Contract: "Two year",
  InternetService: "DSL",
  OnlineSecurity: "Yes",
  TechSupport: "Yes",
  PaymentMethod: "Bank transfer (automatic)",
};

const highRiskSample: PredictPayload = {
  tenure: 2,
  MonthlyCharges: 105,
  TotalCharges: 210,
  SeniorCitizen: 0,
  Partner: "No",
  Contract: "Month-to-month",
  InternetService: "Fiber optic",
  OnlineSecurity: "No",
  TechSupport: "No",
  PaymentMethod: "Electronic check",
};

type ScanHistoryRow = {
  id: number;
  tenure: number;
  monthly: number;
  churn: number;
  revenueAtRisk: number;
  riskLevel: string;
  confidence: number;
  timestamp: string;
};

function currency(value: number): string {
  return new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 2 }).format(value || 0);
}

function percent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function riskBadgeClass(level: string): string {
  if (level.includes("No")) return "badge no";
  if (level.includes("Low")) return "badge low";
  if (level.includes("Medium")) return "badge medium";
  return "badge high";
}

function Gauge({ risk }: { risk: number }) {
  const radius = 92;
  const circumference = 2 * Math.PI * radius;
  const score = Math.round(risk * 100);
  const offset = circumference - (score / 100) * circumference;

  let stroke = "var(--accent)";
  if (score >= 25 && score < 50) stroke = "var(--blue)";
  if (score >= 50 && score < 75) stroke = "var(--amber)";
  if (score >= 75) stroke = "var(--red)";

  return (
    <div className="gauge-wrap">
      <svg viewBox="0 0 240 240" className="gauge-svg">
        <circle cx="120" cy="120" r={radius} className="gauge-base" />
        <motion.circle
          cx="120"
          cy="120"
          r={radius}
          className="gauge-value"
          stroke={stroke}
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset: offset }}
          transition={{ duration: 1.1, ease: "easeOut" }}
          style={{ strokeDasharray: circumference }}
        />
      </svg>
      <div className="gauge-center">
        <strong>{score}</strong>
        <span>risk score</span>
      </div>
    </div>
  );
}

export default function Page() {
  const [customer, setCustomer] = useState<PredictPayload>(highRiskSample);
  const [prediction, setPrediction] = useState<BatchRow | null>(null);
  const [explain, setExplain] = useState<ExplainItem[]>(EMPTY_EXPLAIN);
  const [history, setHistory] = useState<ScanHistoryRow[]>([]);
  const [batchFile, setBatchFile] = useState<File | null>(null);
  const [batchRows, setBatchRows] = useState<BatchRow[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string>("");

  const leaderboard = useMemo(() => [...batchRows].sort((a, b) => b.churn_probability - a.churn_probability).slice(0, 10), [batchRows]);

  const cohortData = useMemo(() => {
    const grouped: Record<string, { contract: string; avg: number; count: number }> = {};
    for (const row of batchRows) {
      const key = row.contract_type || "Unknown";
      if (!grouped[key]) grouped[key] = { contract: key, avg: 0, count: 0 };
      grouped[key].avg += row.churn_probability;
      grouped[key].count += 1;
    }
    return Object.values(grouped).map((group) => ({
      contract: group.contract,
      avg: group.count ? group.avg / group.count : 0,
      count: group.count,
    }));
  }, [batchRows]);

  const summary = useMemo(() => {
    const totalScans = history.length;
    const avgConfidence = totalScans
      ? history.reduce((sum, row) => sum + row.confidence, 0) / totalScans
      : 0;
    const highRiskPercent = totalScans
      ? history.filter((row) => row.riskLevel.includes("High")).length / totalScans
      : 0;
    const totalRevenueAtRisk = history.reduce((sum, row) => sum + row.revenueAtRisk, 0);

    return {
      totalScans,
      avgConfidence,
      highRiskPercent,
      totalRevenueAtRisk,
    };
  }, [history]);

  async function runPrediction() {
    setError("");
    setIsLoading(true);
    try {
      const result = await explainCustomer(customer);
      const nextPrediction: BatchRow = {
        ...result,
        row: 0,
      };
      setPrediction(nextPrediction);
      setExplain(result.shap_waterfall || EMPTY_EXPLAIN);

      setHistory((current) => {
        const row: ScanHistoryRow = {
          id: Date.now(),
          tenure: customer.tenure,
          monthly: customer.MonthlyCharges,
          churn: result.churn_probability,
          revenueAtRisk: result.revenue_at_risk,
          riskLevel: result.risk_level,
          confidence: result.confidence,
          timestamp: new Date().toLocaleTimeString(),
        };
        return [row, ...current].slice(0, 10);
      });
    } catch (caught) {
      setError("Prediction failed. Check backend status and payload fields.");
    } finally {
      setIsLoading(false);
    }
  }

  async function runBatchUpload() {
    if (!batchFile) {
      setError("Select a CSV first.");
      return;
    }

    setError("");
    setIsUploading(true);
    try {
      const batch = await predictBatch(batchFile);
      setBatchRows(batch.predictions || []);
    } catch (caught) {
      setError("Batch scoring failed. Confirm the CSV format and backend availability.");
    } finally {
      setIsUploading(false);
    }
  }

  return (
    <main className="dashboard">
      <header className="topbar">
        <div>
          <h1>⬡ CHURNGUARD AI <span className="live">LIVE</span></h1>
          <p>Predict churn severity, inspect drivers, and monitor portfolio risk in real time.</p>
        </div>
      </header>

      <section className="summary-grid">
        <article><span>Total Scans</span><strong>{summary.totalScans}</strong></article>
        <article><span>Avg Confidence</span><strong>{percent(summary.avgConfidence)}</strong></article>
        <article><span>High Risk %</span><strong>{percent(summary.highRiskPercent)}</strong></article>
        <article><span>Revenue @ Risk</span><strong>{currency(summary.totalRevenueAtRisk)}</strong></article>
      </section>

      {error ? <div className="error-box">{error}</div> : null}

      <section className="panel two-col">
        <article className="card">
          <div className="card-head">
            <h2>Customer Input</h2>
            <div className="presets">
              <button type="button" onClick={() => setCustomer(lowRiskSample)}>Low Risk Sample</button>
              <button type="button" onClick={() => setCustomer(highRiskSample)}>High Risk Sample</button>
            </div>
          </div>

          <div className="form-grid">
            <label>
              Tenure
              <input type="number" value={customer.tenure} onChange={(event) => setCustomer((c) => ({ ...c, tenure: Number(event.target.value) }))} />
            </label>
            <label>
              MonthlyCharges
              <input type="number" value={customer.MonthlyCharges} onChange={(event) => setCustomer((c) => ({ ...c, MonthlyCharges: Number(event.target.value) }))} />
            </label>
            <label>
              TotalCharges
              <input type="number" value={customer.TotalCharges} onChange={(event) => setCustomer((c) => ({ ...c, TotalCharges: Number(event.target.value) }))} />
            </label>
            <label>
              SeniorCitizen
              <div className="toggle-wrap">
                <button type="button" className={customer.SeniorCitizen === 1 ? "toggle active" : "toggle"} onClick={() => setCustomer((c) => ({ ...c, SeniorCitizen: 1 }))}>Yes</button>
                <button type="button" className={customer.SeniorCitizen === 0 ? "toggle active" : "toggle"} onClick={() => setCustomer((c) => ({ ...c, SeniorCitizen: 0 }))}>No</button>
              </div>
            </label>
            <label>
              Partner
              <div className="toggle-wrap">
                <button type="button" className={customer.Partner === "Yes" ? "toggle active" : "toggle"} onClick={() => setCustomer((c) => ({ ...c, Partner: "Yes" }))}>Yes</button>
                <button type="button" className={customer.Partner === "No" ? "toggle active" : "toggle"} onClick={() => setCustomer((c) => ({ ...c, Partner: "No" }))}>No</button>
              </div>
            </label>
            <label>
              OnlineSecurity
              <div className="toggle-wrap">
                <button type="button" className={customer.OnlineSecurity === "Yes" ? "toggle active" : "toggle"} onClick={() => setCustomer((c) => ({ ...c, OnlineSecurity: "Yes" }))}>Yes</button>
                <button type="button" className={customer.OnlineSecurity === "No" ? "toggle active" : "toggle"} onClick={() => setCustomer((c) => ({ ...c, OnlineSecurity: "No" }))}>No</button>
              </div>
            </label>
            <label>
              TechSupport
              <div className="toggle-wrap">
                <button type="button" className={customer.TechSupport === "Yes" ? "toggle active" : "toggle"} onClick={() => setCustomer((c) => ({ ...c, TechSupport: "Yes" }))}>Yes</button>
                <button type="button" className={customer.TechSupport === "No" ? "toggle active" : "toggle"} onClick={() => setCustomer((c) => ({ ...c, TechSupport: "No" }))}>No</button>
              </div>
            </label>
            <label>
              Contract
              <select value={customer.Contract} onChange={(event) => setCustomer((c) => ({ ...c, Contract: event.target.value as PredictPayload["Contract"] }))}>
                <option>Month-to-month</option>
                <option>One year</option>
                <option>Two year</option>
              </select>
            </label>
            <label>
              InternetService
              <select value={customer.InternetService} onChange={(event) => setCustomer((c) => ({ ...c, InternetService: event.target.value as PredictPayload["InternetService"] }))}>
                <option>DSL</option>
                <option>Fiber optic</option>
                <option>No</option>
              </select>
            </label>
            <label>
              PaymentMethod
              <select value={customer.PaymentMethod} onChange={(event) => setCustomer((c) => ({ ...c, PaymentMethod: event.target.value as PredictPayload["PaymentMethod"] }))}>
                <option>Electronic check</option>
                <option>Mailed check</option>
                <option>Bank transfer (automatic)</option>
                <option>Credit card (automatic)</option>
              </select>
            </label>
          </div>

          <button className="analyze" type="button" onClick={runPrediction} disabled={isLoading}>
            ▶ ANALYZE CUSTOMER
          </button>
        </article>

        <article className="card right-stack">
          <h2>Risk Gauge</h2>
          <Gauge risk={prediction?.churn_probability ?? 0} />
          <p className={riskBadgeClass(prediction?.risk_level || "No Risk")}>{prediction?.risk_level || "No Risk"}</p>

          <div className="revenue-meter">
            <h3>Revenue at Risk</h3>
            <strong>{currency(prediction?.revenue_at_risk ?? 0)}</strong>
            <span>Annualized using P(Medium)+P(High)</span>
          </div>
        </article>
      </section>

      <section className="panel two-col">
        <article className="card">
          <div className="card-head">
            <h2>SHAP Waterfall</h2>
            <BoltIcon className="icon" />
          </div>
          <div className="chart-slot">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={explain} layout="vertical" margin={{ top: 8, right: 20, left: 24, bottom: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                <XAxis type="number" tickFormatter={(v) => Number(v).toFixed(2)} />
                <YAxis dataKey="feature" type="category" width={180} />
                <Tooltip
                  formatter={(value: number, _: string, payload: unknown) => {
                    const item = payload as { payload?: ExplainItem };
                    return [Number(value).toFixed(4), `Customer value: ${item.payload?.customer_value ?? 0}`];
                  }}
                />
                <Bar dataKey="value" radius={[6, 6, 6, 6]}>
                  {explain.map((item) => (
                    <Cell key={item.feature} fill={item.value >= 0 ? "#ff6b6b" : "#00e5a0"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </article>

        <article className="card">
          <div className="card-head">
            <h2>Batch CSV Upload</h2>
            <ShieldCheckIcon className="icon" />
          </div>
          <div className="upload-zone">
            <input type="file" accept=".csv" onChange={(event) => setBatchFile(event.target.files?.[0] || null)} />
            <button type="button" onClick={runBatchUpload} disabled={isUploading}>{isUploading ? "Scoring..." : "Score Batch"}</button>
            <p>Total customers scored: {batchRows.length}</p>
          </div>

          <div className="table-wrap">
            <table>
              <thead>
                <tr><th>#</th><th>Risk</th><th>Churn%</th><th>Rev@Risk</th></tr>
              </thead>
              <tbody>
                {leaderboard.map((row, index) => (
                  <tr key={`${row.row}-${index}`}>
                    <td>{index + 1}</td>
                    <td><span className={riskBadgeClass(row.risk_level)}>{row.risk_level}</span></td>
                    <td>{percent(row.churn_probability)}</td>
                    <td>{currency(row.revenue_at_risk)}</td>
                  </tr>
                ))}
                {!leaderboard.length ? <tr><td colSpan={4}>Upload CSV to generate leaderboard.</td></tr> : null}
              </tbody>
            </table>
          </div>
        </article>
      </section>

      <section className="panel two-col">
        <article className="card">
          <div className="card-head">
            <h2>Scan History</h2>
            <ArrowTrendingUpIcon className="icon" />
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>#</th>
                  <th>Tenure</th>
                  <th>Monthly$</th>
                  <th>Risk Level</th>
                  <th>Churn%</th>
                  <th>Rev@Risk</th>
                  <th>Timestamp</th>
                </tr>
              </thead>
              <tbody>
                {history.map((row, index) => (
                  <tr key={row.id}>
                    <td>{index + 1}</td>
                    <td>{row.tenure}</td>
                    <td>{currency(row.monthly)}</td>
                    <td><span className={riskBadgeClass(row.riskLevel)}>{row.riskLevel}</span></td>
                    <td>{percent(row.churn)}</td>
                    <td>{currency(row.revenueAtRisk)}</td>
                    <td>{row.timestamp}</td>
                  </tr>
                ))}
                {!history.length ? <tr><td colSpan={7}>Run predictions to build scan history.</td></tr> : null}
              </tbody>
            </table>
          </div>
        </article>

        <article className="card">
          <h2>Cohort Analysis by Contract Type</h2>
          <div className="chart-slot">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={cohortData}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                <XAxis dataKey="contract" />
                <YAxis tickFormatter={(value) => `${Math.round(value * 100)}%`} />
                <Tooltip formatter={(value: number) => `${(Number(value) * 100).toFixed(1)}%`} />
                <Bar dataKey="avg" fill="#60a5fa" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </article>
      </section>
    </main>
  );
}

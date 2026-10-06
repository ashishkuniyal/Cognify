import { useState, useEffect } from 'react';
import { 
  FlaskConical, BarChart3, MessageSquareText, Sun, Database, 
  BrainCircuit, Users, Target, GraduationCap, Utensils, BookOpen, Lightbulb, AlertTriangle, CheckCircle2
} from 'lucide-react';
import { 
  Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer 
} from 'recharts';
import './index.css';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export default function App() {
  const [activeTab, setActiveTab] = useState('predict');
  const [loading, setLoading] = useState(false);
  const [formData, setFormData] = useState({
    gender: 'female',
    race_ethnicity: 'group C',
    parental_level_of_education: "bachelor's degree",
    lunch: 'standard',
    test_preparation_course: 'none',
    attendance_rate: 85.0,
    study_hours_per_week: 10.0,
    previous_gpa: 3.2,
    assignment_completion_rate: 90.0
  });

  const [prediction, setPrediction] = useState(null);

  const handleChange = (e) => {
    const value = e.target.type === 'number' || e.target.type === 'range' ? parseFloat(e.target.value) : e.target.value;
    setFormData({ ...formData, [e.target.name]: value });
  };

  const handleSubmit = async (e) => {
    if (e) e.preventDefault();
    setLoading(true);

    try {
      const response = await fetch(`${API_URL}/api/v1/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      });
      const data = await response.json();
      setPrediction(data);
    } catch (err) {
      console.error(err);
    }
    setLoading(false);
  };

  const mockChartData = prediction ? [
    { subject: 'Math', A: prediction.math_score || 0, fullMark: 100 },
    { subject: 'Reading', A: prediction.reading_score || 0, fullMark: 100 },
    { subject: 'Writing', A: prediction.writing_score || 0, fullMark: 100 },
    { subject: 'Overall', A: prediction.overall_score || 0, fullMark: 100 },
  ] : [
    { subject: 'Math', A: 73, fullMark: 100 },
    { subject: 'Reading', A: 85, fullMark: 100 },
    { subject: 'Writing', A: 60, fullMark: 100 },
    { subject: 'Overall', A: 72, fullMark: 100 },
  ];

  const getRiskColor = (level) => {
    if (level === 'HIGH') return '#ef4444'; // Red
    if (level === 'MEDIUM') return '#f59e0b'; // Amber
    return '#10b981'; // Green
  };

  return (
    <div className="app-layout">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="brand">
          <BrainCircuit className="brand-icon" size={28} />
          <h2>Cognify</h2>
        </div>
        <nav className="nav-menu">
          <button className={`nav-btn ${activeTab === 'predict' ? 'active' : ''}`} onClick={() => setActiveTab('predict')}>
            <FlaskConical size={18} /> Prediction Engine
          </button>
          <button className={`nav-btn ${activeTab === 'analytics' ? 'active' : ''}`} onClick={() => setActiveTab('analytics')}>
            <BarChart3 size={18} /> Model Monitoring
          </button>
          <button className={`nav-btn ${activeTab === 'chat' ? 'active' : ''}`} onClick={() => setActiveTab('chat')}>
            <MessageSquareText size={18} /> AI Counselor
          </button>
        </nav>
        <div className="sidebar-footer">
          <span className="version">Cognify ML v3.0</span>
          <button className="theme-toggle"><Sun size={18} /></button>
        </div>
      </aside>

      {/* Main Content */}
      <main className="main-content">
        <header className="page-header">
          <div className="header-badge">
            <span className="badge-dot"></span>
            Cognify Enterprise Intelligence
          </div>
          <h1>Academic Risk Intelligence</h1>
          <p>Early-warning system to identify at-risk students, explain predictions, and recommend personalized interventions.</p>
        </header>

        <div className="stats-row">
          <div className="stat-card">
            <div className="stat-icon-wrapper"><Database size={24} className="teal-icon" /></div>
            <div>
              <h3>XGBoost + RF</h3>
              <span>CORE ML ENGINES</span>
            </div>
          </div>
          <div className="stat-card">
            <div className="stat-icon-wrapper"><Target size={24} className="teal-icon" /></div>
            <div>
              <h3>0.67</h3>
              <span>MODEL R² (VALIDATED)</span>
            </div>
          </div>
          <div className="stat-card">
            <div className="stat-icon-wrapper"><Users size={24} className="teal-icon" /></div>
            <div>
              <h3>SHAP</h3>
              <span>GLOBAL EXPLAINABILITY</span>
            </div>
          </div>
        </div>

        {activeTab === 'predict' && (
          <div className="matrix-card">
            <form className="matrix-form scrollable-form" onSubmit={handleSubmit}>
              
              <h3 className="section-title">Demographics & Background</h3>
              <div className="form-row">
                <div className="input-group">
                  <label>GENDER</label>
                  <select name="gender" value={formData.gender} onChange={handleChange}>
                    <option value="male">Male</option>
                    <option value="female">Female</option>
                  </select>
                </div>
                <div className="input-group">
                  <label>ETHNICITY</label>
                  <select name="race_ethnicity" value={formData.race_ethnicity} onChange={handleChange}>
                    <option value="group A">Group A</option>
                    <option value="group B">Group B</option>
                    <option value="group C">Group C</option>
                    <option value="group D">Group D</option>
                    <option value="group E">Group E</option>
                  </select>
                </div>
              </div>
              
              <div className="input-group full-width">
                <label>PARENTAL EDUCATION</label>
                <select name="parental_level_of_education" value={formData.parental_level_of_education} onChange={handleChange}>
                  <option value="some high school">Some High School</option>
                  <option value="high school">High School</option>
                  <option value="some college">Some College</option>
                  <option value="associate's degree">Associate's Degree</option>
                  <option value="bachelor's degree">Bachelor's Degree</option>
                  <option value="master's degree">Master's Degree</option>
                </select>
              </div>

              <div className="form-row">
                <div className="input-group">
                  <label>LUNCH TYPE</label>
                  <select name="lunch" value={formData.lunch} onChange={handleChange}>
                    <option value="standard">Standard</option>
                    <option value="free/reduced">Free/Reduced</option>
                  </select>
                </div>
                <div className="input-group">
                  <label>TEST PREP</label>
                  <select name="test_preparation_course" value={formData.test_preparation_course} onChange={handleChange}>
                    <option value="none">None</option>
                    <option value="completed">Completed</option>
                  </select>
                </div>
              </div>

              <h3 className="section-title" style={{marginTop: '20px'}}>Behavioral & Academic Features (What-If)</h3>
              
              <div className="input-group full-width slider-group">
                <div className="slider-header">
                  <label>ATTENDANCE RATE (%)</label>
                  <span>{formData.attendance_rate}%</span>
                </div>
                <input type="range" name="attendance_rate" min="0" max="100" step="1" value={formData.attendance_rate} onChange={handleChange} />
              </div>

              <div className="input-group full-width slider-group">
                <div className="slider-header">
                  <label>STUDY HOURS / WEEK</label>
                  <span>{formData.study_hours_per_week}h</span>
                </div>
                <input type="range" name="study_hours_per_week" min="0" max="40" step="1" value={formData.study_hours_per_week} onChange={handleChange} />
              </div>

              <div className="input-group full-width slider-group">
                <div className="slider-header">
                  <label>PREVIOUS GPA (1.0 - 4.0)</label>
                  <span>{formData.previous_gpa}</span>
                </div>
                <input type="range" name="previous_gpa" min="1.0" max="4.0" step="0.1" value={formData.previous_gpa} onChange={handleChange} />
              </div>

              <div className="input-group full-width slider-group">
                <div className="slider-header">
                  <label>ASSIGNMENT COMPLETION (%)</label>
                  <span>{formData.assignment_completion_rate}%</span>
                </div>
                <input type="range" name="assignment_completion_rate" min="0" max="100" step="1" value={formData.assignment_completion_rate} onChange={handleChange} />
              </div>

              <button type="submit" className="btn-predict" disabled={loading}>
                {loading ? 'Processing...' : <><FlaskConical size={18} /> Run Prediction Engine</>}
              </button>
            </form>

            <div className="matrix-results">
              {prediction ? (
                <div className="results-wrapper">
                  <div className="results-header">
                    <div>
                      <span className="results-label">Academic Risk Level</span>
                      <div className="cluster-name" style={{ color: getRiskColor(prediction.risk_level) }}>
                        {prediction.risk_level} RISK
                      </div>
                    </div>
                    <div className="base-math">
                      <span className="results-label">Overall Predicted Score</span>
                      <div className="math-score" style={{ color: '#14b8a6' }}>
                        {prediction.overall_score ? prediction.overall_score.toFixed(1) : '--'}
                      </div>
                    </div>
                  </div>
                  
                  <div className="radar-container" style={{height: '220px', marginTop: '-10px'}}>
                    <ResponsiveContainer width="100%" height="100%">
                      <RadarChart cx="50%" cy="50%" outerRadius="70%" data={mockChartData}>
                        <PolarGrid stroke="#1e293b" />
                        <PolarAngleAxis dataKey="subject" tick={{ fill: '#94a3b8', fontSize: 12 }} />
                        <Radar name="Student" dataKey="A" stroke="#0d9488" fill="#14b8a6" fillOpacity={0.3} />
                      </RadarChart>
                    </ResponsiveContainer>
                  </div>

                  <div className="explanation-box" style={{ backgroundColor: '#0f172a', padding: '15px', borderRadius: '10px', marginBottom: '15px', border: '1px solid #1e293b' }}>
                    <h4 style={{ color: '#94a3b8', fontSize: '0.8rem', marginBottom: '10px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <AlertTriangle size={16} color="#f59e0b" /> SHAP EXPLANATION (Top Risk Factor)
                    </h4>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ color: '#f8fafc', fontWeight: '600' }}>{prediction.top_risk_factor}</span>
                      <span style={{ color: prediction.top_risk_impact > 0 ? '#ef4444' : '#10b981', fontWeight: '700' }}>
                        {prediction.top_risk_impact > 0 ? '+' : ''}{prediction.top_risk_impact.toFixed(3)}
                      </span>
                    </div>
                  </div>

                  <div className="advisor-box" style={{ backgroundColor: 'rgba(20, 184, 166, 0.05)', padding: '15px', borderRadius: '10px', border: '1px solid rgba(20, 184, 166, 0.2)' }}>
                    <h4 style={{ color: '#14b8a6', fontSize: '0.8rem', marginBottom: '10px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <CheckCircle2 size={16} /> RECOMMENDED ACTIONS
                    </h4>
                    <p style={{ color: '#e2e8f0', fontSize: '0.9rem', lineHeight: '1.6' }}>
                      {prediction.advisor_message.replace('Recommendations: ', '')}
                    </p>
                  </div>
                  
                  <div style={{ textAlign: 'center', marginTop: '15px' }}>
                    <span style={{ fontSize: '0.7rem', color: '#475569' }}>Predictions are estimates and should support—not replace—academic judgment.</span>
                  </div>
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', color: '#64748b' }}>
                  <FlaskConical size={48} style={{ opacity: 0.2, marginBottom: '16px' }} />
                  <h3>Awaiting Input</h3>
                  <p style={{ textAlign: 'center', fontSize: '0.9rem', maxWidth: '300px', marginTop: '8px' }}>Enter student features and click "Run Prediction Engine" to predict academic risk.</p>
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === 'analytics' && (
          <div className="matrix-card" style={{ padding: '40px', flexDirection: 'column', gap: '20px' }}>
            <h2 style={{ color: '#14b8a6', marginBottom: '20px' }}>Model Monitoring & Analytics</h2>
            <p style={{ color: '#94a3b8' }}>Tracking model performance metrics on the demo dataset.</p>
            <div style={{ display: 'flex', gap: '20px', marginTop: '20px' }}>
              <div style={{ flex: 1, backgroundColor: '#0b1121', padding: '20px', borderRadius: '12px', border: '1px solid #1e293b' }}>
                <span style={{ color: '#94a3b8', fontSize: '0.8rem', fontWeight: '700' }}>REGRESSION PERFORMANCE (R²)</span>
                <h3 style={{ color: '#10b981', fontSize: '1.5rem', marginTop: '8px' }}>0.67</h3>
                <p style={{ color: '#64748b', fontSize: '0.85rem', marginTop: '8px' }}>A realistic performance score. We do not claim 99% accuracy on human behavioral data.</p>
              </div>
              <div style={{ flex: 1, backgroundColor: '#0b1121', padding: '20px', borderRadius: '12px', border: '1px solid #1e293b' }}>
                <span style={{ color: '#94a3b8', fontSize: '0.8rem', fontWeight: '700' }}>DATASET SIZE</span>
                <h3 style={{ color: '#f8fafc', fontSize: '1.5rem', marginTop: '8px' }}>1,000 Students</h3>
                <p style={{ color: '#64748b', fontSize: '0.85rem', marginTop: '8px' }}>Augmented with synthetic behavioral features to demonstrate real-world use case.</p>
              </div>
            </div>
            
            <div style={{ backgroundColor: '#0b1121', padding: '20px', borderRadius: '12px', border: '1px solid #1e293b', marginTop: '20px' }}>
                <span style={{ color: '#94a3b8', fontSize: '0.8rem', fontWeight: '700' }}>DATA DRIFT STATUS</span>
                <p style={{ color: '#64748b', fontSize: '0.85rem', marginTop: '8px' }}>Insufficient historical data for drift analysis.</p>
            </div>
          </div>
        )}

        {activeTab === 'chat' && (
          <div className="matrix-card" style={{ padding: '40px', flexDirection: 'column', height: '500px' }}>
            <h2 style={{ color: '#14b8a6', marginBottom: '20px' }}>AI Counselor Chat</h2>
            <div style={{ flex: 1, backgroundColor: '#0b1121', borderRadius: '12px', border: '1px solid #1e293b', padding: '20px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <div style={{ display: 'flex', gap: '12px', alignItems: 'flex-start' }}>
                <div style={{ backgroundColor: 'rgba(20, 184, 166, 0.1)', padding: '10px', borderRadius: '8px', color: '#14b8a6' }}><BrainCircuit size={20} /></div>
                <div style={{ backgroundColor: '#1e293b', padding: '12px 16px', borderRadius: '12px', color: '#f8fafc', maxWidth: '80%' }}>
                  Hello! I am your AI Counselor powered by Gemini. I can explain the SHAP explanations and help explore what-if scenarios. How can I help today?
                </div>
              </div>
            </div>
            <div style={{ display: 'flex', gap: '12px', marginTop: '20px' }}>
              <input type="text" placeholder="Ask about student interventions..." style={{ flex: 1, backgroundColor: '#0b1121', border: '1px solid #1e293b', padding: '12px 16px', borderRadius: '8px', color: 'white', outline: 'none' }} />
              <button style={{ backgroundColor: '#14b8a6', border: 'none', padding: '0 24px', borderRadius: '8px', color: '#040914', fontWeight: 'bold', cursor: 'pointer' }}>Send</button>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  GraduationCap, Users, BookOpen, Utensils, Target, 
  Sparkles, Database, BrainCircuit, AlertCircle, BarChart3, FlaskConical,
  Sun, Moon, ShieldAlert, CheckCircle2, Compass, AlertTriangle, Lightbulb,
  MessageSquareText, Send
} from 'lucide-react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, AreaChart, Area,
  Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis
} from 'recharts';
import './index.css';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

function App() {
  const [activeTab, setActiveTab] = useState('predict');
  const [theme, setTheme] = useState('dark');
  
  const [formData, setFormData] = useState({
    gender: 'male',
    race_ethnicity: 'group A',
    parental_level_of_education: "bachelor's degree",
    lunch: 'standard',
    test_preparation_course: 'none'
  });

  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  const [analyticsData, setAnalyticsData] = useState([]);
  const [systemHealth, setSystemHealth] = useState({ status: 'healthy', msg: 'No Data Drift Detected' });

  // Chat State
  const [chatMessages, setChatMessages] = useState([
    { role: 'ai', text: "Hello! I am your AI Counselor. Please run a prediction first so I can analyze your profile, then ask me anything about your career or academic future!" }
  ]);
  const [chatInput, setChatInput] = useState('');
  const [chatLoading, setChatLoading] = useState(false);
  const chatEndRef = useRef(null);

  // Theme Toggle Effect
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  // Fetch Analytics & Calculate Drift
  useEffect(() => {
    if (activeTab === 'analytics') {
      fetch(`${API_URL}/api/analytics`)
        .then(res => res.json())
        .then(data => {
          setAnalyticsData(data.reverse());
          
          if (data.length > 10) {
            const avgScore = data.reduce((acc, curr) => acc + curr.predicted_math_score, 0) / data.length;
            if (avgScore < 45 || avgScore > 85) {
              setSystemHealth({ status: 'warning', msg: `Data Drift Warning: Avg Score is ${avgScore.toFixed(1)}` });
            } else {
              setSystemHealth({ status: 'healthy', msg: 'System Normal: No Data Drift' });
            }
          }
        })
        .catch(err => console.error("Failed to load analytics", err));
    }
  }, [activeTab]);

  // Auto-scroll chat
  useEffect(() => {
    if (activeTab === 'chat' && chatEndRef.current) {
      chatEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [chatMessages, activeTab]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const response = await fetch(`${API_URL}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Failed to predict');

      setTimeout(() => {
        setPrediction(data);
        setLoading(false);
        // Add context to chat
        setChatMessages(prev => [
          ...prev, 
          { role: 'ai', text: `I've just analyzed your profile. It looks like you have strong potential in the ${data.career_recommendation} path. What would you like to know?` }
        ]);
      }, 500);

    } catch (err) {
      setError(err.message);
      setLoading(false);
    }
  };

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!chatInput.trim()) return;

    const userMsg = chatInput;
    setChatInput('');
    setChatMessages(prev => [...prev, { role: 'user', text: userMsg }]);
    setChatLoading(true);

    try {
      const response = await fetch(`${API_URL}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',   // Required for server-side session cookies
        body: JSON.stringify({
          message: userMsg,
          context: prediction
        })
      });
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.error || "Failed to connect to LLM");
      }
      
      setChatMessages(prev => [...prev, { role: 'ai', text: data.reply }]);
    } catch (err) {
      setChatMessages(prev => [...prev, { role: 'ai', text: `System Error: ${err.message}` }]);
    } finally {
      setChatLoading(false);
    }
  };

  const radarData = prediction ? [
    { subject: 'Math', score: prediction.math_score },
    { subject: 'Reading', score: prediction.reading_score },
    { subject: 'Writing', score: prediction.writing_score }
  ] : [];

  return (
    <>
      <div className="bg-mesh"></div>
      
      <main className="app-wrapper">
        
        {/* Left Hero Section */}
        <motion.div className="hero-section" initial={{ opacity: 0, x: -50 }} animate={{ opacity: 1, x: 0 }}>
          <div className="badge">
            <Sparkles size={16} />
            <span>Student Matrix V2 AI</span>
          </div>
          <h1 className="hero-title">Intelligent EdTech Platform</h1>
          <p className="hero-subtitle">
            A multi-engine Machine Learning architecture predicting full academic potential, career trajectories, and identifying at-risk students instantly.
          </p>
          <div className="stats-grid">
            <motion.div className="stat-card" whileHover={{ y: -5 }}>
              <Database size={24} color={theme === 'dark' ? "#6366f1" : "#4f46e5"} />
              <div className="stat-value" style={{color: 'var(--text-main)'}}>3 AI</div>
              <div className="stat-label">Active ML Engines</div>
            </motion.div>
            <motion.div className="stat-card" whileHover={{ y: -5 }}>
              <Compass size={24} color="#ec4899" />
              <div className="stat-value" style={{color: 'var(--text-main)'}}>K-Means</div>
              <div className="stat-label">Career Clustering</div>
            </motion.div>
          </div>
        </motion.div>

        {/* Right Application Section */}
        <motion.div className="form-section" initial={{ opacity: 0, x: 50 }} animate={{ opacity: 1, x: 0 }}>
          <div className="glass-panel">
            
            {/* Tabs & Theme Toggle */}
            <div className="tabs">
              <div className="tab-group">
                <button className={`tab-btn ${activeTab === 'predict' ? 'active' : ''}`} onClick={() => setActiveTab('predict')}>
                  <FlaskConical size={18} /> Prediction
                </button>
                <button className={`tab-btn ${activeTab === 'analytics' ? 'active' : ''}`} onClick={() => setActiveTab('analytics')}>
                  <BarChart3 size={18} /> Analytics
                </button>
                <button className={`tab-btn ${activeTab === 'chat' ? 'active' : ''}`} onClick={() => setActiveTab('chat')}>
                  <MessageSquareText size={18} /> AI Counselor
                </button>
              </div>
              <button className="theme-toggle" onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')} title="Toggle Light/Dark Mode">
                {theme === 'dark' ? <Sun size={20} /> : <Moon size={20} />}
              </button>
            </div>

            <AnimatePresence mode="wait">
              {/* Prediction Tab */}
              {activeTab === 'predict' && (
                <motion.div key="predict" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                  <form className="form-grid" onSubmit={handleSubmit}>
                    <div className="form-group">
                      <label className="input-label">Gender</label>
                      <div className="input-wrapper">
                        <Users size={18} className="input-icon" />
                        <select name="gender" value={formData.gender} onChange={handleChange} className="custom-input">
                          <option value="male">Male</option>
                          <option value="female">Female</option>
                        </select>
                      </div>
                    </div>

                    <div className="form-group">
                      <label className="input-label">Ethnicity</label>
                      <div className="input-wrapper">
                        <Target size={18} className="input-icon" />
                        <select name="race_ethnicity" value={formData.race_ethnicity} onChange={handleChange} className="custom-input">
                          <option value="group A">Group A</option>
                          <option value="group B">Group B</option>
                          <option value="group C">Group C</option>
                          <option value="group D">Group D</option>
                          <option value="group E">Group E</option>
                        </select>
                      </div>
                    </div>

                    <div className="form-group full">
                      <label className="input-label">Parental Education</label>
                      <div className="input-wrapper">
                        <GraduationCap size={18} className="input-icon" />
                        <select name="parental_level_of_education" value={formData.parental_level_of_education} onChange={handleChange} className="custom-input">
                          <option value="some high school">Some High School</option>
                          <option value="high school">High School</option>
                          <option value="some college">Some College</option>
                          <option value="associate's degree">Associate's Degree</option>
                          <option value="bachelor's degree">Bachelor's Degree</option>
                          <option value="master's degree">Master's Degree</option>
                        </select>
                      </div>
                    </div>

                    <div className="form-group">
                      <label className="input-label">Lunch Type</label>
                      <div className="input-wrapper">
                        <Utensils size={18} className="input-icon" />
                        <select name="lunch" value={formData.lunch} onChange={handleChange} className="custom-input">
                          <option value="standard">Standard</option>
                          <option value="free/reduced">Free/Reduced</option>
                        </select>
                      </div>
                    </div>

                    <div className="form-group">
                      <label className="input-label">Test Prep Course</label>
                      <div className="input-wrapper">
                        <BookOpen size={18} className="input-icon" />
                        <select name="test_preparation_course" value={formData.test_preparation_course} onChange={handleChange} className="custom-input">
                          <option value="none">None</option>
                          <option value="completed">Completed</option>
                        </select>
                      </div>
                    </div>

                    <button type="submit" className="submit-btn" disabled={loading}>
                      {loading ? (
                        <motion.div animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 1, ease: "linear" }}>
                          <BrainCircuit size={20} />
                        </motion.div>
                      ) : (
                        <BrainCircuit size={20} />
                      )}
                      {loading ? 'Analyzing Neural Matrix...' : 'Run Prediction Model'}
                    </button>

                    {error && (
                      <div className="error-card">
                        <AlertCircle size={20} /> <span>{error}</span>
                      </div>
                    )}
                  </form>

                  {/* V2 Results Output */}
                  <AnimatePresence>
                    {prediction !== null && !error && (
                      <motion.div className="result-wrapper" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.5 }}>
                        
                        {/* At-Risk Warning */}
                        {prediction.is_at_risk && (
                          <div className="stat-card" style={{background: 'rgba(239, 68, 68, 0.1)', borderColor: 'rgba(239, 68, 68, 0.5)'}}>
                            <div style={{display: 'flex', alignItems: 'center', gap: '1rem'}}>
                              <AlertTriangle size={32} color="#ef4444" />
                              <div>
                                <div style={{color: '#ef4444', fontWeight: '700', fontSize: '1.1rem'}}>AT-RISK STUDENT DETECTED</div>
                                <div style={{color: 'var(--text-muted)', fontSize: '0.9rem'}}>This profile indicates a high probability of scoring below average.</div>
                              </div>
                            </div>
                          </div>
                        )}

                        {/* Radar Chart & Career */}
                        <div className="result-card" style={{padding: '1rem'}}>
                          <div style={{display: 'flex', justifyContent: 'space-between', width: '100%', alignItems: 'center', padding: '0 1rem'}}>
                            <div>
                              <div className="result-label">Recommended Path</div>
                              <div style={{color: 'var(--primary)', fontWeight: '700', fontSize: '1.2rem'}}>{prediction.career_recommendation}</div>
                            </div>
                            <div style={{textAlign: 'right'}}>
                              <div className="result-label">Math Score</div>
                              <div style={{color: 'var(--text-main)', fontWeight: '700', fontSize: '2rem'}}>{Math.round(prediction.math_score)}</div>
                            </div>
                          </div>
                          
                          <div style={{width: '100%', height: '250px'}}>
                            <ResponsiveContainer width="100%" height="100%">
                              <RadarChart cx="50%" cy="50%" outerRadius="70%" data={radarData}>
                                <PolarGrid stroke="var(--glass-border)" />
                                <PolarAngleAxis dataKey="subject" tick={{fill: 'var(--text-muted)', fontSize: 12}} />
                                <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
                                <Radar name="Score" dataKey="score" stroke={theme === 'dark' ? "#818cf8" : "#4f46e5"} fill={theme === 'dark' ? "#818cf8" : "#4f46e5"} fillOpacity={0.5} />
                              </RadarChart>
                            </ResponsiveContainer>
                          </div>
                        </div>

                        {/* Prescriptive AI Advisor */}
                        <div className="explain-card" style={{marginTop: '0'}}>
                          <div className="explain-header" style={{marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem'}}>
                            <Lightbulb size={18} color="#eab308" />
                            <span style={{color: 'var(--text-main)', fontWeight: '600'}}>AI Prescriptive Advisor</span>
                          </div>
                          <p style={{color: 'var(--text-muted)', fontSize: '0.9rem', fontStyle: 'italic', lineHeight: '1.5'}}>
                            "{prediction.advisor_message}"
                          </p>
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </motion.div>
              )}

              {/* Analytics Tab */}
              {activeTab === 'analytics' && (
                <motion.div key="analytics" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="analytics-container">
                  
                  {/* MLOps System Health Card */}
                  <div className="stat-card" style={{marginBottom: '1.5rem', background: systemHealth.status === 'healthy' ? 'rgba(34, 197, 94, 0.1)' : 'rgba(239, 68, 68, 0.1)'}}>
                    <div style={{display: 'flex', alignItems: 'center', gap: '1rem'}}>
                      {systemHealth.status === 'healthy' ? <CheckCircle2 size={32} color="#22c55e" /> : <ShieldAlert size={32} color="#ef4444" />}
                      <div>
                        <div style={{color: 'var(--text-main)', fontWeight: '600'}}>System Health</div>
                        <div style={{color: 'var(--text-muted)', fontSize: '0.85rem'}}>{systemHealth.msg}</div>
                      </div>
                    </div>
                  </div>

                  <h3 className="analytics-title" style={{color: 'var(--text-main)'}}>Live Inference Stream</h3>
                  <div className="chart-wrapper">
                    {analyticsData.length > 0 ? (
                      <ResponsiveContainer width="100%" height={250}>
                        <AreaChart data={analyticsData}>
                          <defs>
                            <linearGradient id="colorScore" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%" stopColor={theme === 'dark' ? "#6366f1" : "#4f46e5"} stopOpacity={0.8}/>
                              <stop offset="95%" stopColor={theme === 'dark' ? "#6366f1" : "#4f46e5"} stopOpacity={0}/>
                            </linearGradient>
                          </defs>
                          <CartesianGrid strokeDasharray="3 3" stroke="rgba(150,150,150,0.1)" vertical={false} />
                          <XAxis dataKey="id" stroke="var(--text-muted)" fontSize={12} tickLine={false} axisLine={false} />
                          <YAxis stroke="var(--text-muted)" fontSize={12} tickLine={false} axisLine={false} domain={['dataMin - 10', 'dataMax + 10']} />
                          <RechartsTooltip contentStyle={{ backgroundColor: 'var(--panel-bg)', borderColor: 'var(--glass-border)', borderRadius: '8px', color: 'var(--text-main)' }} />
                          <Area type="monotone" dataKey="predicted_math_score" stroke={theme === 'dark' ? "#818cf8" : "#4f46e5"} strokeWidth={3} fillOpacity={1} fill="url(#colorScore)" />
                        </AreaChart>
                      </ResponsiveContainer>
                    ) : (
                      <div className="empty-state" style={{color: 'var(--text-muted)'}}>No predictions logged yet.</div>
                    )}
                  </div>
                </motion.div>
              )}

              {/* Chat Tab */}
              {activeTab === 'chat' && (
                <motion.div key="chat" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} style={{display: 'flex', flexDirection: 'column', height: '500px'}}>
                  
                  {/* Chat Header */}
                  <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem'}}>
                    <div style={{display: 'flex', alignItems: 'center', gap: '0.5rem'}}>
                      <div style={{width: '8px', height: '8px', borderRadius: '50%', background: '#22c55e', boxShadow: '0 0 6px #22c55e'}}></div>
                      <span style={{color: 'var(--text-muted)', fontSize: '0.85rem'}}>Gemini 3.8 Flash · Multi-turn Memory</span>
                    </div>
                    <button
                      onClick={async () => {
                        await fetch(`${API_URL}/api/chat/reset`, { method: 'POST', credentials: 'include' });
                        setChatMessages([{ role: 'ai', text: "Chat cleared! Run a new prediction and let's start fresh." }]);
                      }}
                      style={{background: 'none', border: '1px solid var(--glass-border)', color: 'var(--text-muted)', padding: '0.25rem 0.75rem', borderRadius: '0.5rem', cursor: 'pointer', fontSize: '0.8rem'}}
                    >
                      Clear
                    </button>
                  </div>

                  {/* Messages area */}
                  <div className="chat-messages" style={{flex: 1}}>
                    {chatMessages.length === 0 && (
                      <div style={{textAlign: 'center', color: 'var(--text-muted)', padding: '2rem', fontSize: '0.9rem'}}>
                        <MessageSquareText size={40} style={{margin: '0 auto 1rem', opacity: 0.3}} />
                        <p>Run a prediction first, then ask me anything!</p>
                      </div>
                    )}
                    {chatMessages.map((msg, i) => (
                      <div key={i} className={`chat-bubble ${msg.role}`}>
                        {msg.text}
                      </div>
                    ))}
                    {chatLoading && (
                      <div className="chat-bubble ai">
                        <motion.span animate={{ opacity: [0.3, 1, 0.3] }} transition={{ repeat: Infinity, duration: 1.2 }}>
                          ● ● ●
                        </motion.span>
                      </div>
                    )}
                    <div ref={chatEndRef} />
                  </div>

                  {/* Suggested prompts — only show when idle and prediction loaded */}
                  {prediction && !chatLoading && chatMessages.length <= 1 && (
                    <div style={{display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '0.75rem'}}>
                      {["What career suits me best?", "How can I improve my scores?", "Am I at risk of failing?"].map(q => (
                        <button key={q} onClick={() => { setChatInput(q); }}
                          style={{background: 'var(--glass)', border: '1px solid var(--glass-border)', color: 'var(--text-muted)', padding: '0.4rem 0.75rem', borderRadius: '2rem', cursor: 'pointer', fontSize: '0.78rem', transition: '0.2s'}}
                          onMouseEnter={e => e.target.style.borderColor = 'var(--primary)'}
                          onMouseLeave={e => e.target.style.borderColor = 'var(--glass-border)'}
                        >{q}</button>
                      ))}
                    </div>
                  )}
                  
                  <form className="chat-input-wrapper" onSubmit={handleSendMessage}>
                    <input 
                      type="text" 
                      className="chat-input" 
                      placeholder={prediction ? "Ask about your career, scores, study tips..." : "Run a prediction first..."}
                      value={chatInput}
                      onChange={(e) => setChatInput(e.target.value)}
                      disabled={chatLoading}
                    />
                    <button type="submit" className="chat-send-btn" disabled={chatLoading || !chatInput.trim()}>
                      <Send size={18} />
                    </button>
                  </form>
                </motion.div>
              )}

            </AnimatePresence>

          </div>
        </motion.div>
      </main>
    </>
  );
}

export default App;

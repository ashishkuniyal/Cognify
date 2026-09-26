import { useState } from 'react';
import './index.css';

function App() {
  const [formData, setFormData] = useState({
    gender: 'male',
    race_ethnicity: 'group A',
    parental_level_of_education: "bachelor's degree",
    lunch: 'standard',
    test_preparation_course: 'none',
    reading_score: 50,
    writing_score: 50
  });

  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const response = await fetch('http://localhost:5000/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(formData)
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to predict');
      }

      setPrediction(data.prediction);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <div className="glass-card">
        <header className="header">
          <h1 className="title">Student Matrix</h1>
          <p className="subtitle">Predict academic performance with machine learning</p>
        </header>

        <form onSubmit={handleSubmit} className="form-grid">
          <div className="form-group">
            <label>Gender</label>
            <select name="gender" value={formData.gender} onChange={handleChange} required>
              <option value="male">Male</option>
              <option value="female">Female</option>
            </select>
          </div>

          <div className="form-group">
            <label>Race / Ethnicity</label>
            <select name="race_ethnicity" value={formData.race_ethnicity} onChange={handleChange} required>
              <option value="group A">Group A</option>
              <option value="group B">Group B</option>
              <option value="group C">Group C</option>
              <option value="group D">Group D</option>
              <option value="group E">Group E</option>
            </select>
          </div>

          <div className="form-group full-width">
            <label>Parental Education</label>
            <select name="parental_level_of_education" value={formData.parental_level_of_education} onChange={handleChange} required>
              <option value="some high school">Some High School</option>
              <option value="high school">High School</option>
              <option value="some college">Some College</option>
              <option value="associate's degree">Associate's Degree</option>
              <option value="bachelor's degree">Bachelor's Degree</option>
              <option value="master's degree">Master's Degree</option>
            </select>
          </div>

          <div className="form-group">
            <label>Lunch Type</label>
            <select name="lunch" value={formData.lunch} onChange={handleChange} required>
              <option value="standard">Standard</option>
              <option value="free/reduced">Free/Reduced</option>
            </select>
          </div>

          <div className="form-group">
            <label>Test Prep Course</label>
            <select name="test_preparation_course" value={formData.test_preparation_course} onChange={handleChange} required>
              <option value="none">None</option>
              <option value="completed">Completed</option>
            </select>
          </div>

          <div className="form-group">
            <label>Reading Score</label>
            <input 
              type="number" 
              name="reading_score" 
              value={formData.reading_score} 
              onChange={handleChange} 
              min="0" max="100" 
              required 
            />
          </div>

          <div className="form-group">
            <label>Writing Score</label>
            <input 
              type="number" 
              name="writing_score" 
              value={formData.writing_score} 
              onChange={handleChange} 
              min="0" max="100" 
              required 
            />
          </div>

          <button type="submit" className="submit-btn" disabled={loading}>
            {loading ? 'Analyzing Data...' : 'Predict Math Score'}
          </button>

          {error && (
            <div className="error-message">
              {error}
            </div>
          )}

          {prediction !== null && (
            <div className="result-container">
              <div className="result-label">Predicted Math Score</div>
              <div className="result-value">{prediction.toFixed(2)}</div>
            </div>
          )}
        </form>
      </div>
    </div>
  );
}

export default App;

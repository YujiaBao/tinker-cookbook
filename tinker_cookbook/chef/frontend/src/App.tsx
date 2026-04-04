import { BrowserRouter, Link, Route, Routes, useLocation } from 'react-router-dom';
import { RunListPage } from './pages/RunListPage';
import { RunDetailPage } from './pages/RunDetailPage';
import { RolloutDetailPage } from './pages/RolloutDetailPage';
import { EvalPage } from './pages/EvalPage';
import { EvalRunDetailPage } from './pages/EvalRunDetailPage';
import { EvalTrajectoryPage } from './pages/EvalTrajectoryPage';
import './App.css';

function Nav() {
  const location = useLocation();
  const isTraining = location.pathname === '/' || location.pathname.startsWith('/runs');
  const isEval = location.pathname.startsWith('/eval');

  return (
    <header className="app-header">
      <Link to="/" className="app-logo">Tinker Chef</Link>
      <nav style={{ display: 'flex', gap: '0.5rem', marginLeft: '1.5rem' }}>
        <Link
          to="/"
          className={`tab ${isTraining ? 'active' : ''}`}
          style={{ borderBottom: 'none', padding: '0.25rem 0.75rem' }}
        >
          Training
        </Link>
        <Link
          to="/eval"
          className={`tab ${isEval ? 'active' : ''}`}
          style={{ borderBottom: 'none', padding: '0.25rem 0.75rem' }}
        >
          Eval
        </Link>
      </nav>
      <span className="app-tagline" style={{ marginLeft: 'auto' }}>Training Visualization Dashboard</span>
    </header>
  );
}

function App() {
  return (
    <BrowserRouter>
      <div className="app">
        <Nav />
        <main className="app-main">
          <Routes>
            <Route path="/" element={<RunListPage />} />
            <Route path="/runs/:runId" element={<RunDetailPage />} />
            <Route path="/runs/:runId/iterations/:iteration/rollouts/:groupIdx/:trajIdx" element={<RolloutDetailPage />} />
            <Route path="/eval" element={<EvalPage />} />
            <Route path="/eval/:evalRunId" element={<EvalRunDetailPage />} />
            <Route path="/eval/:evalRunId/:benchmark/:idx" element={<EvalTrajectoryPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}

export default App;

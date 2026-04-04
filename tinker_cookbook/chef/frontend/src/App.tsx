import { BrowserRouter, Link, Route, Routes } from 'react-router-dom';
import { DashboardPage } from './pages/DashboardPage';
import { RunDetailPage } from './pages/RunDetailPage';
import { RolloutDetailPage } from './pages/RolloutDetailPage';
import { CompareRunsPage } from './pages/CompareRunsPage';
import { EvalRunDetailPage } from './pages/EvalRunDetailPage';
import { EvalTrajectoryPage } from './pages/EvalTrajectoryPage';
import './App.css';

function App() {
  return (
    <BrowserRouter>
      <div className="app">
        <header className="app-header">
          <Link to="/" className="app-logo">Tinker Chef</Link>
          <span className="app-tagline">Training Dashboard</span>
        </header>
        <main className="app-main">
          <Routes>
            <Route path="/" element={<DashboardPage />} />
            <Route path="/compare" element={<CompareRunsPage />} />
            <Route path="/runs/:runId" element={<RunDetailPage />} />
            <Route path="/runs/:runId/iterations/:iteration/rollouts/:groupIdx/:trajIdx" element={<RolloutDetailPage />} />
            <Route path="/eval/:evalRunId" element={<EvalRunDetailPage />} />
            <Route path="/eval/:evalRunId/:benchmark/:idx" element={<EvalTrajectoryPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}

export default App;

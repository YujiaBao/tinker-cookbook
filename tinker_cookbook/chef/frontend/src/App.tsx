import { BrowserRouter, Route, Routes } from 'react-router-dom';
import { RunListPage } from './pages/RunListPage';
import { RunDetailPage } from './pages/RunDetailPage';
import { RolloutDetailPage } from './pages/RolloutDetailPage';
import './App.css';

function App() {
  return (
    <BrowserRouter>
      <div className="app">
        <header className="app-header">
          <a href="/" className="app-logo">Tinker Chef</a>
          <span className="app-tagline">Training Visualization Dashboard</span>
        </header>
        <main className="app-main">
          <Routes>
            <Route path="/" element={<RunListPage />} />
            <Route path="/runs/:runId" element={<RunDetailPage />} />
            <Route path="/runs/:runId/iterations/:iteration/rollouts/:groupIdx/:trajIdx" element={<RolloutDetailPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}

export default App;
